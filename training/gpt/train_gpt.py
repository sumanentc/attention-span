"""
train_gpt.py

From-scratch GPT-124M pretraining on the TinyStories dataset.

Consolidates every optimization built up incrementally:
  - Data pipeline: tokenize -> concat -> sliding-window chunk (input_ids / target_ids)
  - Architecture: weight tying, GPT-2 style init (+ NANOGPT_SCALE_INIT residual scaling,
    credit: Andrej Karpathy's nanoGPT), FlashAttention (F.scaled_dot_product_attention)
  - Precision: TF32 matmul + bfloat16 autocast mixed precision
  - Optimizer: configure_optimizers (decay/no-decay param groups) + fused AdamW
  - Training loop: cosine LR schedule w/ warmup, gradient clipping, gradient accumulation,
    CUDA-event based throughput timing, periodic eval + sample generation
  - torch.compile for kernel fusion
  - Vocab size rounded to 50304 (nanoGPT trick) for tensor-core-friendly shapes

Usage:
    # Colab (single GPU, in a notebook cell or script mode)
    !python train_gpt.py --num_epochs 1 --micro_batch_size 16 --grad_accum_steps 4

    # VS Code / remote GPU cluster (terminal)
    python train_gpt.py --num_epochs 1 --micro_batch_size 16 --grad_accum_steps 4 \
        --eval_freq 500 --eval_iter 50 --checkpoint_dir ./checkpoints
"""

import os

# Suppress TensorFlow/oneDNN INFO logs. These are triggered indirectly — the
# `datasets`/`transformers` libraries check for a TensorFlow backend at import
# time, and if TensorFlow happens to be installed (e.g. Colab's default image),
# it prints its own startup logs even though we never use TensorFlow ourselves.
# Must be set before datasets/transformers (and therefore TensorFlow) are imported.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import time
import math
import inspect
import argparse
import warnings
from itertools import chain
from importlib.metadata import version

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

DATASET_NAME = "roneneldan/TinyStories"


# =============================================================================
# Environment setup
# =============================================================================

def setup_environment(seed: int = 1337):
    """Reproducibility, device selection, and TF32 matmul precision."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        capability = torch.cuda.get_device_capability()
        # Enables TF32 for float32 matmuls on Ampere+ GPUs (Volta 7.0+, Turing 7.5+,
        # Ampere 8.0+, Hopper 9.0+). NOTE: does not convert tensors to bf16 — this
        # only changes the internal precision tensor cores use for the matmul's
        # multiply-accumulate step. Actual bf16 training is handled separately via
        # torch.autocast in the training loop.
        if capability[0] >= 7:
            torch.set_float32_matmul_precision("high")
            print("TF32 enabled (tensor cores in use for float32 matmuls)")
        else:
            print("Tensor cores not supported on this GPU. Using default precision.")

    try:
        print(f"torch version: {version('torch')}")
        print(f"tiktoken version: {version('tiktoken')}")
        print(f"transformers version: {version('transformers')}")
        print(f"datasets version: {version('datasets')}")
        print(f"safetensors version: {version('safetensors')}")
    except Exception:
        pass

    return device


# =============================================================================
# Memory tracking helpers
# =============================================================================

def start_memory_tracking():
    """Initialize GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("This script is intended for CUDA GPUs but CUDA is not available.")


def print_memory_usage():
    if not torch.cuda.is_available():
        print("This script is intended for CUDA GPUs but CUDA is not available.")
        return
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"Device Name: {device_name}")
    print(f"Total Memory available: {total_memory / (1024**3):.2f} GB")
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")


def cleanup(device):
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    torch.cuda.reset_peak_memory_stats()
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")


# =============================================================================
# Data pipeline: tokenize -> concat -> chunk
# =============================================================================

def make_tokenize_function(tokenizer):
    def tokenize_function(sample):
        """
        Tokenizes a batch of raw text samples.
        - truncation=True + max_length=tokenizer.model_max_length: caps any single
          sample at the tokenizer's max length (1024 for GPT-2).
        - No padding: padding before concatenation would insert pad tokens into
          the middle of the flattened token stream, corrupting the corpus that
          chunk() slices into fixed-length training windows.
        - Appends tokenizer.eos_token_id to the end of every sample's tokens.
          Since concat() later joins all stories into one continuous stream with
          no other separator, this gives the model an explicit, learnable signal
          for "a new, unrelated document starts here" — matching how GPT-2 itself
          was trained (documents joined via <|endoftext|>), rather than silently
          blending story boundaries together.
        """
        tokenized = tokenizer(
            text=sample["text"],
            truncation=True,
            max_length=tokenizer.model_max_length - 1  # reserve room for the appended eos token
        )
        tokenized["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in tokenized["input_ids"]]
        tokenized["attention_mask"] = [mask + [1] for mask in tokenized["attention_mask"]]
        return tokenized
    return tokenize_function


def concat(examples):
    """Flattens a batch of tokenized sequences into one continuous stream of tokens."""
    examples["input_ids"] = list(chain.from_iterable(examples["input_ids"]))
    examples["attention_mask"] = list(chain.from_iterable(examples["attention_mask"]))
    return examples


def make_chunk_function(tokenizer):
    tokenize_function = make_tokenize_function(tokenizer)

    def chunk(examples, max_length, stride):
        """
        Converts a batch of raw text samples into fixed-length (input, target) chunks
        for next-token-prediction pretraining, using a sliding window.
        """
        tokenized_ds = tokenize_function(examples)
        concated_ds = concat(tokenized_ds)
        input_ids = concated_ds["input_ids"]

        input_ids_truncated = []
        target_ids_truncated = []

        for i in range(0, len(input_ids) - max_length, stride):
            input_chunk = input_ids[i:i + max_length]
            target_chunk = input_ids[i + 1: i + max_length + 1]  # shifted by 1 position
            input_ids_truncated.append(input_chunk)
            target_ids_truncated.append(target_chunk)

        examples["input_ids"] = input_ids_truncated
        examples["target_ids"] = target_ids_truncated
        return examples
    return chunk


def build_dataloaders(tokenizer, cfg, batch_size, nprocs):
    """Loads the full TinyStories dataset and builds train/val DataLoaders."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, cache_dir="tiny_stories")
    chunk_fn = make_chunk_function(tokenizer)

    train_ds = dataset["train"]
    tokenized_train = train_ds.map(
        chunk_fn, batched=True, batch_size=batch_size, num_proc=nprocs,
        remove_columns=["text"],
        fn_kwargs={"max_length": cfg["context_length"], "stride": cfg["context_length"]}
    )
    tokenized_train.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_train, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=nprocs, pin_memory=True
    )

    val_ds = dataset["validation"]
    tokenized_val = val_ds.map(
        chunk_fn, batched=True, batch_size=batch_size, num_proc=nprocs,
        remove_columns=["text"],
        fn_kwargs={"max_length": cfg["context_length"], "stride": cfg["context_length"]}
    )
    tokenized_val.set_format("torch")
    val_dataloader = DataLoader(
        tokenized_val, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=nprocs, pin_memory=True
    )

    print(f"Train data loader: 1 epoch consists of {len(train_dataloader)} batches")
    print(f"Validation data loader: 1 epoch consists of {len(val_dataloader)} batches")

    return train_dataloader, val_dataloader


def count_total_tokens(train_dataloader, val_dataloader):
    """Counts total input tokens across the full train/val DataLoaders (one pass each)."""
    train_tokens = 0
    for batch in train_dataloader:
        input_batch = batch["input_ids"]
        train_tokens += input_batch.numel()

    val_tokens = 0
    for batch in val_dataloader:
        input_batch = batch["input_ids"]
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("Total tokens:", train_tokens + val_tokens)

    return train_tokens, val_tokens


# =============================================================================
# Model architecture
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        # nanoGPT convention (credit: Andrej Karpathy) — flags this layer for
        # scaled-down init in GPTModel._init_weights (residual-stream output projection).
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch's fused scaled_dot_product_attention
    (dispatches to FlashAttention on supported hardware)."""

    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.out_proj.NANOGPT_SCALE_INIT = 1

        self.dropout = dropout  # plain float, passed directly into SDPA

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)        # [b, num_tokens, d_out]
        queries = self.W_query(x)   # [b, num_tokens, d_out]
        values = self.W_value(x)    # [b, num_tokens, d_out]

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Fused FlashAttention kernel — never materializes the full
        # [b, num_heads, num_tokens, num_tokens] attention score matrix in memory.
        context_vec = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=True
        )

        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying (GPT-2 architecture): reduces params from ~163M to ~124M
        self.tok_emb.weight = self.out_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        GPT-2 / nanoGPT style init (credit: Andrej Karpathy's nanoGPT for the
        NANOGPT_SCALE_INIT residual scaling scheme).
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.cfg["n_layers"]) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        GPT-3-style AdamW parameter grouping: weight decay only on 2D+ parameters
        (Linear/Embedding weight matrices); biases and LayerNorm scale/shift are
        excluded from decay. Uses fused AdamW when available on CUDA.
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# =============================================================================
# Generation & loss helper functions
# =============================================================================

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy autoregressive generation."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text_sampled(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    Sampling-based autoregressive generation, as an alternative to greedy decoding.

    Unlike generate_text_simple (which always picks the single highest-probability
    token, and is therefore fully deterministic — identical prompts always produce
    identical output), this introduces controlled randomness:

      - temperature: scales the logits before softmax. temperature < 1.0 sharpens
        the distribution (more confident/greedy-like); > 1.0 flattens it (more
        random/diverse). temperature=1.0 leaves the distribution unchanged.
        temperature=0.0 is handled as a special case, falling back to greedy
        argmax decoding (equivalent to generate_text_simple) rather than
        dividing by zero.
      - top_k: if set, restricts sampling to only the top_k highest-probability
        tokens at each step (discarding the long unlikely tail), then samples
        from among those. Set to None to sample from the full vocabulary.

    This helps address repeated/looping output on similar or identical prompts,
    at the cost of losing full determinism (two runs with the same prompt can
    now produce different results, unless a manual seed is fixed beforehand).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            min_value = top_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, torch.tensor(-float("inf")).to(logits.device), logits)

        if temperature == 0.0:
            # temperature -> 0 is mathematically equivalent to greedy decoding
            # (softmax becomes one-hot on the argmax), but dividing by 0 directly
            # would produce inf/nan and crash torch.multinomial. Handle explicitly.
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap torch.compile
    context_size = raw_model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))

    model.train()


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    logits = logits.view(-1, logits.size(-1))
    loss = F.cross_entropy(logits, target_batch.view(-1))
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i < num_batches:
            input_batch = batch["input_ids"]
            target_batch = batch["target_ids"]
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, num_batches):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=num_batches)
    model.train()
    return train_loss, val_loss


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss", save_path=None):
    """
    Plots training vs. validation metrics (loss, accuracy, etc.) against epochs,
    with a secondary x-axis for examples/tokens seen. Generic and reusable across
    metrics via the `label` param.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)  # invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved {label} plot to: {save_path}")
    plt.close(fig)


# =============================================================================
# Cosine learning rate schedule with warmup
# =============================================================================

def get_lr(it, max_steps, warmup_steps, max_lr=6e-4):
    """
    Cosine learning rate schedule with linear warmup.
    Credit: nanoGPT / GPT-3 style LR schedule.
    """
    min_lr = max_lr * 0.1

    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# Training loop (mixed precision + gradient accumulation + cosine LR + clipping)
# =============================================================================

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                 eval_freq, eval_iter, start_context, tokenizer, max_new_tokens,
                 warmup_steps=10, max_lr=6e-4, grad_accum_steps=1, checkpoint_dir=None,
                 keep_last_n_checkpoints=3):
    train_losses, val_losses, track_tokens, track_epochs = [], [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = micro_batches_per_epoch // grad_accum_steps
    num_training_steps = num_epochs * steps_per_epoch
    device_type = device.type
    print(f"micro-batches/epoch: {micro_batches_per_epoch} | "
          f"optimizer steps/epoch: {steps_per_epoch} | total optimizer steps: {num_training_steps}")

    cumulative_tokens, cumulative_time = 0.0, 0.0
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_start.record()
    else:
        t0 = time.time()

    progress_bar = tqdm(total=num_training_steps)

    for epoch in range(num_epochs):
        model.train()
        print_memory_usage()

        optimizer.zero_grad()
        micro_step = 0

        for batch in train_loader:
            input_batch = batch["input_ids"]
            target_batch = batch["target_ids"]

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / grad_accum_steps

            loss.backward()

            total_tokens += input_batch.numel()
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                global_step += 1

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                lr = get_lr(global_step, num_training_steps, warmup_steps, max_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)

                if global_step % eval_freq == 0 or global_step == num_training_steps - 1:
                    if use_cuda:
                        t_end.record()
                        torch.cuda.synchronize()
                        elapsed = t_start.elapsed_time(t_end) / 1000
                        t_start.record()
                    else:
                        elapsed = time.time() - t0
                        t0 = time.time()

                    tokens_interval = total_tokens - last_tokens
                    last_tokens = total_tokens
                    tps = tokens_interval / elapsed if elapsed > 0 else 0

                    if global_step:
                        cumulative_tokens += tokens_interval
                        cumulative_time += elapsed

                    avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens.append(total_tokens)
                    track_epochs.append(global_step / steps_per_epoch)  # fractional epoch progress

                    print(f"Ep {epoch+1}, Step {global_step:06d} | Train loss: {train_loss:.3f} | "
                          f"Val loss: {val_loss:.3f} | norm: {norm:.4f} | LR: {lr:.6f} | "
                          f"Step tok/sec: {round(tps)} | Avg tok/sec: {round(avg_tps)}")

                    generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens)
                    print_memory_usage()

        # Save a checkpoint once per epoch (not tied to eval_freq) — keeps disk usage
        # bounded and predictable regardless of how often we evaluate.
        if checkpoint_dir is not None:
            save_checkpoint(model, optimizer, epoch + 1, global_step, checkpoint_dir,
                             keep_last_n=keep_last_n_checkpoints)

    return train_losses, val_losses, track_tokens, track_epochs


def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, keep_last_n=3):
    """
    Saves a checkpoint named by epoch (e.g. ckpt_epoch001.pt). Keeps only the most
    recent `keep_last_n` epoch checkpoints on disk, deleting older ones automatically.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap torch.compile
    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch:03d}.pt")
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Keep only the most recent `keep_last_n` epoch checkpoints, delete older ones
    existing = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_epoch") and f.endswith(".pt")
    )
    if len(existing) > keep_last_n:
        for old_ckpt in existing[:-keep_last_n]:
            os.remove(os.path.join(checkpoint_dir, old_ckpt))
            print(f"Removed old checkpoint: {old_ckpt}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train a from-scratch GPT-124M model on TinyStories")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=16,
                         help="Batch size that actually fits in GPU memory per forward/backward pass")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                         help="Number of micro-batches to accumulate gradients over before each optimizer step")
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--eval_iter", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--start_context", type=str, default="Once upon a time")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                         help="Number of most recent epoch checkpoints to retain on disk")
    parser.add_argument("--compile", action="store_true", default=True,
                         help="Use torch.compile for kernel fusion")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--nprocs", type=int, default=None,
                         help="Number of processes for dataset tokenization (default: half of CPU count)")
    args = parser.parse_args()

    # ---- Environment setup ----
    device = setup_environment()

    # ---- Config ----
    GPT_CONFIG_124M = {
        "vocab_size": 50304,      # rounded up from 50257 to nearest multiple of 64 (nanoGPT trick)
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    nprocs = args.nprocs if args.nprocs is not None else max(1, os.cpu_count() // 2)
    print(f"micro_batch_size: {args.micro_batch_size} | grad_accum_steps: {args.grad_accum_steps} | nprocs: {nprocs}")
    print(GPT_CONFIG_124M)

    # ---- Tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Data ----
    train_dataloader, val_dataloader = build_dataloaders(
        tokenizer, GPT_CONFIG_124M, batch_size=args.micro_batch_size, nprocs=nprocs
    )
    train_tokens, val_tokens = count_total_tokens(train_dataloader, val_dataloader)

    # Quick sanity check: confirm batch shapes look right before committing to a full run
    sample_batch = next(iter(train_dataloader))
    print("Sample batch input_ids shape:", sample_batch["input_ids"].shape)
    print("Sample batch target_ids shape:", sample_batch["target_ids"].shape)

    # ---- Model ----
    cleanup(device)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    device_type = device.type
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        device_type=device_type
    )

    if args.compile:
        model = torch.compile(model)

    start_memory_tracking()

    # ---- Train ----
    training_start_time = time.time()

    train_losses, val_losses, tokens_seen, epochs_seen = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        grad_accum_steps=args.grad_accum_steps,
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints
    )

    cleanup(device)

    # ---- Plot training/validation loss ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    plot_path = os.path.join(args.checkpoint_dir, "loss_plot.png")
    plot_values(epochs_seen, tokens_seen, train_losses, val_losses, label="loss", save_path=plot_path)

    # Note: train_model already saves a checkpoint at the end of every epoch,
    # including the final one — no separate final save needed here.
    total_training_time = time.time() - training_start_time
    hours, rem = divmod(total_training_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Training complete.")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total training tokens: {train_tokens:,}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s "
          f"({total_training_time:.1f} seconds)")
    print(f"Final train loss: {train_losses[-1]:.3f} | Final val loss: {val_losses[-1]:.3f}")


if __name__ == "__main__":
    main()
