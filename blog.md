# Adding Mistral Support to mini-sglang

> **TL;DR:** I added Mistral-7B support to mini-sglang, a lightweight LLM inference engine. The key challenges were handling a subtle Python `None` bug in config parsing and implementing sliding-window attention. Tested on an H100 with FlashAttention-3, the implementation is numerically consistent with HuggingFace within expected tolerance, and sliding window is behaviorally verified with a 6000+ token test.

- **Code:** [github.com/JayZenith/mini-sglang](https://github.com/JayZenith/mini-sglang)
- **Benchmarks:** [github.com/JayZenith/mini-sglang](https://github.com/JayZenith/mini-sglang/compare_logits.py)

## Why Mistral?

Mistral introduced **sliding-window attention**, a form of recency bias where each token only attends to the last N tokens (4096 for Mistral-7B).

Combined with **RoPE** (Rotary Position Embeddings), which encodes relative position by rotating Q/K vectors, Mistral handles long contexts efficiently while maintaining quality.

## The Architecture Insight

Here's the key insight that made this implementation straightforward:

> **Mistral is a Llama-like transformer** (RMSNorm, RoPE, SwiGLU, GQA). The key behavioral difference is sliding-window attention.

Both share:
- RMSNorm (instead of LayerNorm)
- RoPE positional embeddings
- SwiGLU activation (gate × up projection)
- Grouped Query Attention (GQA) with 8 KV heads

This means we can reuse the existing `LlamaForCausalLM` class—just wire up the sliding window config to the attention backend.

---

## Implementation

### Step 1: Config Changes

**File:** `python/minisgl/models/config.py`

Added the `sliding_window` field to `ModelConfig`:

```python
@dataclass(frozen=True)
class ModelConfig:
    # ... existing fields ...
    sliding_window: int | None  # new: Mistral uses 4096
```

And extracted it from HuggingFace config:

```python
sliding_window = getattr(config, "sliding_window", None)
```

### Step 2: Model Registration

**File:** `python/minisgl/models/__init__.py`

Added Mistral to the model factory:

```python
def create_model(model_path: str, model_config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()

    if "llama" in model_name:
        return LlamaForCausalLM(model_config)
    elif "mistral" in model_name:
        return LlamaForCausalLM(model_config)  # Llama-like architecture
    elif "qwen3" in model_name:
        return Qwen3ForCausalLM(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_path}")
```

### Step 3: FlashAttention-3 Backend (Sliding Window)

**File:** `python/minisgl/attention/fa.py`

FlashAttention-3 via sgl-kernel supports `window_size`. I wired it up:

```python
class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config, kvcache, page_table):
        # ...
        self.sliding_window = config.sliding_window

    def forward(self, q, k, v, layer_id, batch):
        # (4096, 0) = look back 4096 tokens, look forward 0 (causal)
        window_size = (self.sliding_window, 0) if self.sliding_window else (-1, -1)

        return _fa_sgl_impl(
            # other args
            window_size=window_size
        )
```

### Step 4: FlashInfer Fallback

**File:** `python/minisgl/attention/fi.py`

FlashInfer doesn't support sliding window, so I added a warning:

```python
if getattr(config, "sliding_window", None):
    logger.warning(
        "Mistral sliding window detected but not supported in FlashInfer. "
        "Defaulting to full attention."
    )
```

---

## The `None` Bug 

After making these changes, I got this cryptic error:

```
TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'
```

The crash happened in linear layer initialization where `head_dim` was `None`.

### The Root Cause

The original config parsing used:

```python
head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
```

This *looks* correct, if `head_dim` doesn't exist, compute it. But here's the trap:

> **In Mistral's HuggingFace config, `head_dim` EXISTS but is set to `None`.**

The `getattr()` fallback only triggers when the attribute is **missing**, not when it's explicitly `None`. So we got `None` instead of `128`.

### The Fix

```python
# Wrong: getattr fallback doesn't trigger for explicit None
head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

# Correct: explicitly handle None
head_dim = getattr(config, "head_dim", None)
if head_dim is None:
    head_dim = config.hidden_size // config.num_attention_heads
```

**Lesson learned:** `getattr(obj, attr, default)` won't save you from explicitly-set `None` values. Always check for `None` explicitly when the fallback matters.

---

## Validation: Proving Correctness

To verify the implementation produces consistent outputs with HuggingFace, I created a **logit comparison test**.

### Test Harness Design

I used a two-phase harness for simplicity and determinism:
- **Phase 1 (`--hf`):** Load HuggingFace model, save reference logits to disk
- **Phase 2 (`--engine`):** Load mini-sglang model, compare against saved reference + sliding window divergence test

> **Note:** The H100's 80GB VRAM could easily co-resident both ~15GB models simultaneously. I kept the two-phase approach because I originally developed it on a 24GB RTX 3090, and it provides cleaner isolation between frameworks for debugging.

### Short Sequence Results (within 4096 window)

```
Test: "The capital of France is" (6 tokens)

Engine vs HuggingFace:
  Max Diff:  0.011719
  Mean Diff: 0.001822
  Top-1 Match: True
  Engine: 'a' (0.1459) vs HF: 'a' (0.1458)
```

Short sequences match tightly—exactly what we expect when sliding window doesn't affect the attention pattern.

---

## Sliding Window Behavioral Verification

Short sequences don't exercise sliding window. To prove it's **actually working**, I tested with sequences **beyond the 4096 token window**.

### Long Sequence Test (7202 tokens)

```
Sequence length: 7202 tokens (exceeds 4096 window by 3106)
Comparison: Last-position logits from full prefill pass
Note: Both HF and engine run with sliding_window=4096 enabled; full-attn run explicitly disables windowing.

--- Windowed Engine vs HuggingFace ---
  Max Diff:  0.192383
  Top-1 Match: True
  Top-1: 'The' vs 'The'

--- Full Attention Engine vs HuggingFace ---
  Max Diff:  4.687500
  Top-1 Match: True
  Top-1: 'The' vs 'The'

--- Windowed vs Full Attention (DIVERGENCE TEST) ---
  Max Diff:  4.796875
```

### Why Does Max Diff Jump from 0.01 to 0.2?

Short sequences stay within the 4096 window, so both implementations compute identical attention patterns. At >4k tokens, the effective attention graph changes (windowing kicks in) and kernel paths diverge more—different masking implementations, different numerical accumulation order. Absolute differences grow even when the behavioral constraint matches.

We treat max_abs_diff ≤ 0.25 as acceptable for long-context FP16 comparisons across different attention kernels.

### What This Proves

1. **Windowed engine is 24x closer to HF** than full attention (0.19 vs 4.69 max diff)
2. **Windowed and full attention diverge significantly** (4.80 max diff), proving the window is actually limiting attention span
3. **Both HF and windowed engine respect sliding window**; the 0.19 diff is expected numerical variation between different attention kernels

### Concrete Artifacts (the receipt)

```
[FA3 BACKEND] window_size=(4096, 0), sliding_window=4096   # windowed
[FA3 BACKEND] window_size=(-1, -1), sliding_window=None    # full attention
```

The backend logs exactly which mode is active. No ambiguity.

<!-- ---

## Key Takeaways

### 1. Architecture Reuse Pays Off
Mistral works with minimal changes to the model forward pass—just config and attention backend tweaks. Understanding model architectures pays off.

### 2. `getattr()` Has a Subtle Trap
```python
# This doesn't protect against explicit None!
value = getattr(obj, "attr", fallback)
```
Always check for `None` explicitly when the fallback matters.

### 3. Know Your Attention Backends
- **FlashAttention-3** (sgl-kernel): Supports sliding window, requires SM90+ (Hopper/H100)
- **FlashInfer**: Broader hardware support, but no sliding window

### 4. Validate Behaviorally, Not Just Numerically
Short sequence tests aren't enough for sliding window. Test with sequences **beyond the window size** and verify:
- Windowed vs full attention diverges (proves window is active)
- Windowed engine tracks reference implementation

### 5. Be Precise About Claims
With different kernels, backends, and dtype paths, you can safely claim:
- **Numerically consistent within tolerance** (max_abs_diff ≤ 0.1 for short sequences, ≤ 0.25 for long-context cross-kernel comparisons)
- **Matching top-1/top-k for tested prompts**
- **Functional equivalence for the tested regime**
- **Sliding window behaviorally verified** (windowed diverges from full attention)

Avoid absolutist phrasing like "mathematically identical" without exhaustive coverage.

### 6. Provide Concrete Artifacts
- Log the actual `window_size` tuple from the kernel call
- Show divergence metrics between windowed/full attention
- Specify tolerance criteria explicitly (e.g., max_abs_diff ≤ 0.1 in FP16)

--- -->

## Resources

- **Code:** [github.com/JayZenith/mini-sglang](https://github.com/JayZenith/mini-sglang)
- **Mistral HF Config:** [huggingface.co/mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- **FlashAttention:** [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **FlashInfer:** [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **mini-sglang:** [github.com/sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)

---

## Appendix: Environment

For reproducibility:

```
GPU: NVIDIA H100 80GB HBM3 (SM90)
Python: 3.12.3
PyTorch: 2.9.1+cu128
CUDA: 12.8
cuDNN: 91002
sgl-kernel: 0.3.21
FlashInfer: 0.5.3
Transformers: 4.57.3
```

See `exact_env.txt` in the repo for the complete package list.
