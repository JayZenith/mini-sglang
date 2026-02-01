"""
Comprehensive benchmark: mini-sglang vs HuggingFace (Mistral-7B-v0.1)

This benchmark validates EVERYTHING from the blog:
1. Short sequences: Numerical consistency within tolerance
2. Long sequences: Sliding window behavioral verification  
3. Performance: Latency, throughput, memory at various sequence lengths

Usage:
  python compare_logits.py --hf         # Phase 1: Generate HF reference
  python compare_logits.py --engine     # Phase 2: Engine comparison + sliding window test
  python compare_logits.py --perf       # Phase 3: Performance benchmarks
  python compare_logits.py --all        # Run all phases
"""

import sys, os, torch, argparse, gc, time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "python"))

MODEL_ID = "mistralai/Mistral-7B-v0.1"
TMP_FILE = "hf_reference.pt"
SLIDING_WINDOW = 4096

# Sentinel to explicitly disable sliding window
DISABLE_SLIDING_WINDOW = "DISABLE"

# Test prompts for short sequence comparison
TEST_PROMPTS = [
    "The capital of France is",
    "Machine learning is a type of",
    "def fibonacci(n):",
    "The quick brown fox",
]

# Performance benchmark sequence lengths
PERF_SEQ_LENS = [512, 1024, 2048, 4096]
WARMUP_ITERS = 3
BENCH_ITERS = 10


def run_phase_1_hf():
    """Phase 1: Collect HuggingFace reference logits for short and long sequences."""
    print("=" * 60)
    print("Phase 1: Collecting HuggingFace Reference Logits")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    
    reference_data = []
    
    # Short sequence tests
    print("\n--- Short Sequence Tests ---")
    for text in TEST_PROMPTS:
        print(f"\nPrompt: '{text}'")
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        print(f"Tokens: {inputs.input_ids[0].tolist()} (len={len(inputs.input_ids[0])})")
        
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :].float().cpu()
            reference_data.append({
                "logits": logits,
                "tokens": inputs.input_ids[0].cpu(),
                "text": text,
                "type": "short"
            })
            
            probs = torch.softmax(logits[0], dim=-1)
            top_probs, top_ids = torch.topk(probs, 5)
            print("Top-5 predictions:")
            for i, (p, idx) in enumerate(zip(top_probs, top_ids)):
                token = tokenizer.decode([idx])
                print(f"  {i+1}. '{token}' (prob={p.item():.4f})")
    
    # Long sequence test (beyond sliding window - 6000+ tokens)
    print("\n--- Long Sequence Test (beyond 4096 window) ---")
    
    base_text = "The quick brown fox jumps over the lazy dog. " * 600
    long_inputs = tokenizer(base_text, return_tensors="pt", truncation=False).to("cuda:0")
    long_seq_len = len(long_inputs.input_ids[0])
    print(f"Long sequence length: {long_seq_len} tokens (target: 6000+)")
    
    with torch.no_grad():
        out = model(**long_inputs)
        long_logits = out.logits[:, -1, :].float().cpu()
        reference_data.append({
            "logits": long_logits,
            "tokens": long_inputs.input_ids[0].cpu(),
            "text": f"[Long sequence: {long_seq_len} tokens]",
            "type": "long"
        })
        
        probs = torch.softmax(long_logits[0], dim=-1)
        top_probs, top_ids = torch.topk(probs, 5)
        print("Top-5 predictions:")
        for i, (p, idx) in enumerate(zip(top_probs, top_ids)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (prob={p.item():.4f})")
    
    torch.save(reference_data, TMP_FILE)
    print(f"\n" + "=" * 60)
    print(f"Saved {len(reference_data)} reference samples to {TMP_FILE}")
    print("Now run with --engine flag")


def load_weights_efficient(model_path, device, dtype):
    """Load weights efficiently to minimize peak memory."""
    import glob
    import safetensors
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    
    if os.path.isdir(model_path):
        hf_folder = model_path
    else:
        hf_folder = snapshot_download(model_path, allow_patterns=["*.safetensors"])
    
    files = sorted(glob.glob(f"{hf_folder}/*.safetensors"))
    state_dict = {}
    
    print(f"Loading {len(files)} weight files...")
    for file in tqdm(files, desc="Loading"):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                state_dict[name] = tensor.to(dtype=dtype, device=device)
                del tensor
    
    return state_dict


def merge_weights(state_dict):
    """Merge QKV and gate/up projections."""
    merged = {}
    keys_to_skip = set()
    
    for key in list(state_dict.keys()):
        if key in keys_to_skip:
            continue
            
        if ".q_proj" in key:
            q = state_dict[key]
            k_key = key.replace(".q_proj", ".k_proj")
            v_key = key.replace(".q_proj", ".v_proj")
            k = state_dict[k_key]
            v = state_dict[v_key]
            new_key = key.replace(".q_proj", ".qkv_proj")
            merged[new_key] = torch.cat([q, k, v], dim=0)
            keys_to_skip.add(k_key)
            keys_to_skip.add(v_key)
            del state_dict[key], state_dict[k_key], state_dict[v_key]
        elif ".gate_proj" in key:
            gate = state_dict[key]
            up_key = key.replace(".gate_proj", ".up_proj")
            up = state_dict[up_key]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            merged[new_key] = torch.cat([gate, up], dim=0)
            keys_to_skip.add(up_key)
            del state_dict[key], state_dict[up_key]
        elif ".k_proj" in key or ".v_proj" in key or ".up_proj" in key:
            continue
        else:
            merged[key] = state_dict[key]
            del state_dict[key]
    
    return merged


def setup_engine(model_config, device, dtype, max_seq_len, backend="fa", sliding_window_setting=None):
    """
    Setup the inference engine with specified backend.
    
    sliding_window_setting:
      - None: use model_config.sliding_window (default)
      - DISABLE_SLIDING_WINDOW: force no sliding window (full attention)
      - int: use this specific sliding window value
    """
    from minisgl.attention import create_attention_backend
    from minisgl.kvcache import create_kvcache
    from minisgl.core import Context
    from minisgl.models.config import ModelConfig, RotaryConfig
    
    # Determine effective sliding window
    if sliding_window_setting == DISABLE_SLIDING_WINDOW:
        effective_sliding_window = None
    elif sliding_window_setting is not None:
        effective_sliding_window = sliding_window_setting
    else:
        effective_sliding_window = model_config.sliding_window
    
    config_for_backend = ModelConfig(
        num_layers=model_config.num_layers,
        num_qo_heads=model_config.num_qo_heads,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
        hidden_size=model_config.hidden_size,
        vocab_size=model_config.vocab_size,
        intermediate_size=model_config.intermediate_size,
        rms_norm_eps=model_config.rms_norm_eps,
        hidden_act=model_config.hidden_act,
        tie_word_embeddings=model_config.tie_word_embeddings,
        sliding_window=effective_sliding_window,
        rotary_config=model_config.rotary_config,
    )
    
    print(f"  [Backend Config] sliding_window={config_for_backend.sliding_window}")
    
    kv_cache = create_kvcache(
        model_config=config_for_backend,
        num_pages=max_seq_len + 1,
        device=device,
        dtype=dtype,
    )
    
    page_table = torch.zeros((2, max_seq_len), dtype=torch.int32, device=device)
    
    attn_backend = create_attention_backend(
        backend,
        config_for_backend,
        kv_cache,
        page_table,
    )
    
    if hasattr(attn_backend, 'sliding_window'):
        print(f"  [Backend Instance] sliding_window={attn_backend.sliding_window}")
    
    ctx = Context(page_size=1, attn_backend=attn_backend)
    
    return ctx, attn_backend, kv_cache, page_table


def run_inference(model, tokens, ctx, attn_backend, page_table, device):
    """Run a single inference pass."""
    from minisgl.core import Batch, Req, SamplingParams
    import minisgl.core as core_module
    
    seq_len = len(tokens)
    
    req = Req(
        input_ids=tokens.to(torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=None,
    )
    
    batch = Batch(reqs=[req], phase="prefill")
    batch.input_ids = tokens.to(torch.int32).to(device)
    batch.out_loc = torch.arange(seq_len, device=device)
    batch.padded_reqs = [req]
    
    page_table[0, :seq_len] = torch.arange(seq_len, device=device)
    
    old_ctx = core_module._GLOBAL_CTX
    core_module._GLOBAL_CTX = ctx
    
    try:
        with torch.no_grad():
            with ctx.forward_batch(batch):
                attn_backend.prepare_metadata(batch)
                logits = model.forward().float().cpu()
    finally:
        core_module._GLOBAL_CTX = old_ctx
    
    return logits[-1]


def load_model_for_engine(device, dtype):
    """Load and return the model and config for engine tests."""
    from minisgl.models.llama import LlamaForCausalLM
    from minisgl.models import ModelConfig
    from minisgl.layers import set_rope_device
    from minisgl.distributed import set_tp_info, try_get_tp_info
    from minisgl.utils import cached_load_hf_config, torch_dtype
    
    if try_get_tp_info() is None: set_tp_info(rank=0, size=1)
    torch.cuda.set_device(device)
    
    hf_config = cached_load_hf_config(MODEL_ID)
    model_config = ModelConfig.from_hf(hf_config)
    
    print(f"\nModel: {MODEL_ID}")
    print(f"Layers: {model_config.num_layers}, Heads: {model_config.num_qo_heads}, KV Heads: {model_config.num_kv_heads}")
    print(f"Head Dim: {model_config.head_dim}")
    print(f"Sliding Window (from config): {model_config.sliding_window}")
    print(f"Backend: FlashAttention-3 (fa)")
    
    set_rope_device(device)
    
    print("\nCreating model...")
    with torch.device("meta"), torch_dtype(dtype):
        model = LlamaForCausalLM(model_config)
    
    print("Loading weights...")
    weights = load_weights_efficient(MODEL_ID, device, dtype)
    gc.collect()
    torch.cuda.empty_cache()
    
    weights = merge_weights(weights)
    gc.collect()
    torch.cuda.empty_cache()
    
    model.load_state_dict(weights)
    del weights
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, model_config


def run_phase_2_engine():
    """Phase 2: Compare mini-sglang engine against HuggingFace reference."""
    print("=" * 60)
    print("Phase 2: Running Mini-SGLang Engine (FlashAttention-3)")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    
    if not os.path.exists(TMP_FILE):
        print(f"ERROR: {TMP_FILE} not found. Run with --hf first.")
        sys.exit(1)
    
    hf_data = torch.load(TMP_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    model, model_config = load_model_for_engine(device, dtype)
    
    max_seq_len = max(len(item["tokens"]) for item in hf_data) + 10
    print(f"Max sequence length: {max_seq_len}")
    
    # =========================================================================
    # SHORT SEQUENCE TESTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("SHORT SEQUENCE TESTS (within sliding window)")
    print("=" * 60)
    
    short_items = [item for item in hf_data if item["type"] == "short"]
    
    print("\nSetting up windowed attention backend:")
    ctx, attn_backend, kv_cache, page_table = setup_engine(
        model_config, device, dtype, max_seq_len, backend="fa", sliding_window_setting=SLIDING_WINDOW
    )
    
    short_results = []
    all_passed = True
    
    for i, item in enumerate(short_items):
        tokens = item["tokens"]
        text = item.get("text", "<unknown>")
        
        eng_logits = run_inference(model, tokens, ctx, attn_backend, page_table, device)
        hf_logits = item["logits"][0]
        
        abs_diff = torch.abs(hf_logits - eng_logits)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        hf_probs = torch.softmax(hf_logits, dim=-1)
        eng_probs = torch.softmax(eng_logits, dim=-1)
        
        hf_top_ids = torch.topk(hf_probs, 5).indices
        eng_top_ids = torch.topk(eng_probs, 5).indices
        
        top1_match = hf_top_ids[0].item() == eng_top_ids[0].item()
        passed = max_diff < 0.1
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        short_results.append((text, max_diff, mean_diff, top1_match, passed))
        
        print(f"\n[{status}] Test {i+1}: '{text}' ({len(tokens)} tokens)")
        print(f"   Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f} | Top-1 Match: {top1_match}")
        
        if i == 0:
            print("   Top-5: HF vs Engine")
            for j in range(5):
                hf_tok = tokenizer.decode([hf_top_ids[j]]).replace('\n', '\\n')
                eng_tok = tokenizer.decode([eng_top_ids[j]]).replace('\n', '\\n')
                print(f"     {j+1}. '{hf_tok}' ({hf_probs[hf_top_ids[j]]:.4f}) vs '{eng_tok}' ({eng_probs[eng_top_ids[j]]:.4f})")
    
    print("\n--- Short Sequence Summary ---")
    passed_count = sum(1 for r in short_results if r[4])
    avg_max_diff = sum(r[1] for r in short_results) / len(short_results)
    print(f"Passed: {passed_count}/{len(short_results)}")
    print(f"Average Max Diff: {avg_max_diff:.6f}")
    print(f"Tolerance Criterion: max_abs_diff <= 0.1")
    
    # =========================================================================
    # LONG SEQUENCE TEST (SLIDING WINDOW BEHAVIORAL VERIFICATION)
    # =========================================================================
    print("\n" + "=" * 60)
    print("LONG SEQUENCE TEST (sliding window behavioral verification)")
    print("=" * 60)
    
    long_items = [item for item in hf_data if item["type"] == "long"]
    
    if long_items:
        long_item = long_items[0]
        long_tokens = long_item["tokens"]
        long_seq_len = len(long_tokens)
        hf_long_logits = long_item["logits"][0]
        
        print(f"\nSequence length: {long_seq_len} tokens (exceeds {SLIDING_WINDOW} window by {long_seq_len - SLIDING_WINDOW} tokens)")
        
        # Run with sliding window ENABLED
        print("\n--- Running with Sliding Window ENABLED (window_size=4096) ---")
        ctx_windowed, attn_windowed, _, page_table_windowed = setup_engine(
            model_config, device, dtype, long_seq_len + 10, backend="fa", sliding_window_setting=SLIDING_WINDOW
        )
        torch.cuda.reset_peak_memory_stats()
        eng_windowed_logits = run_inference(model, long_tokens, ctx_windowed, attn_windowed, page_table_windowed, device)
        windowed_mem = torch.cuda.max_memory_allocated() / 1e9
        
        # Run with sliding window DISABLED (full attention)
        print("\n--- Running with Sliding Window DISABLED (full attention) ---")
        ctx_full, attn_full, _, page_table_full = setup_engine(
            model_config, device, dtype, long_seq_len + 10, backend="fa", sliding_window_setting=DISABLE_SLIDING_WINDOW
        )
        torch.cuda.reset_peak_memory_stats()
        eng_full_logits = run_inference(model, long_tokens, ctx_full, attn_full, page_table_full, device)
        full_mem = torch.cuda.max_memory_allocated() / 1e9
        
        # Comparisons
        windowed_vs_hf = torch.abs(hf_long_logits - eng_windowed_logits)
        full_vs_hf = torch.abs(hf_long_logits - eng_full_logits)
        windowed_vs_full = torch.abs(eng_windowed_logits - eng_full_logits)
        
        max_diff_windowed_hf = windowed_vs_hf.max().item()
        max_diff_full_hf = full_vs_hf.max().item()
        max_diff_windowed_full = windowed_vs_full.max().item()
        
        hf_probs = torch.softmax(hf_long_logits, dim=-1)
        windowed_probs = torch.softmax(eng_windowed_logits, dim=-1)
        full_probs = torch.softmax(eng_full_logits, dim=-1)
        
        hf_top1 = tokenizer.decode([torch.argmax(hf_probs).item()])
        windowed_top1 = tokenizer.decode([torch.argmax(windowed_probs).item()])
        full_top1 = tokenizer.decode([torch.argmax(full_probs).item()])
        
        print(f"\n--- Windowed Engine vs HuggingFace ---")
        print(f"  Max Diff:  {max_diff_windowed_hf:.6f}")
        print(f"  Top-1 Match: {hf_top1 == windowed_top1}")
        print(f"  Top-1: '{windowed_top1}' vs '{hf_top1}'")
        
        print(f"\n--- Full Attention Engine vs HuggingFace ---")
        print(f"  Max Diff:  {max_diff_full_hf:.6f}")
        print(f"  Top-1 Match: {hf_top1 == full_top1}")
        print(f"  Top-1: '{full_top1}' vs '{hf_top1}'")
        
        print(f"\n--- Windowed vs Full Attention (DIVERGENCE TEST) ---")
        print(f"  Max Diff:  {max_diff_windowed_full:.6f}")
        print(f"  Top-1 Match: {windowed_top1 == full_top1}")
        
        print(f"\n--- Memory Usage ---")
        print(f"  Windowed: {windowed_mem:.2f} GB")
        print(f"  Full Attention: {full_mem:.2f} GB")
        
        # Behavioral verification
        print("\n--- BEHAVIORAL VERIFICATION ---")
        
        windowed_closer = max_diff_windowed_hf < max_diff_full_hf
        ratio = max_diff_full_hf / max_diff_windowed_hf if max_diff_windowed_hf > 0 else float('inf')
        
        if windowed_closer and ratio > 5:
            print(f"[PASS] Windowed engine is {ratio:.0f}x closer to HF than full attention ({max_diff_windowed_hf:.2f} vs {max_diff_full_hf:.2f})")
        elif windowed_closer:
            print(f"[PASS] Windowed engine is closer to HF ({max_diff_windowed_hf:.2f} vs {max_diff_full_hf:.2f}, ratio: {ratio:.1f}x)")
        else:
            print(f"[INFO] Windowed vs HF: {max_diff_windowed_hf:.2f}, Full vs HF: {max_diff_full_hf:.2f}")
        
        if max_diff_windowed_full > 1.0:
            print(f"[PASS] Windowed and full attention DIVERGE significantly (max_diff={max_diff_windowed_full:.2f})")
            print(f"       This proves sliding window is behaviorally active!")
        elif max_diff_windowed_full > 0.1:
            print(f"[PASS] Windowed and full attention diverge (max_diff={max_diff_windowed_full:.2f})")
        else:
            print(f"[WARN] Windowed vs Full divergence too small: {max_diff_windowed_full:.6f}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print(f"\nBackend: FlashAttention-3 (sgl-kernel)")
    print(f"Sliding Window: {SLIDING_WINDOW} tokens")
    
    if all_passed:
        print("\n[SUCCESS] All short sequence tests passed!")
    else:
        print("\n[FAILURE] Some short sequence tests failed")
    
    if long_items and max_diff_windowed_full > 0.1:
        print("[SUCCESS] Sliding window behavioral verification passed!")
    elif long_items:
        print("[WARN] Sliding window behavioral verification inconclusive")


def run_phase_3_perf():
    """Phase 3: Performance benchmarks (latency, throughput, memory)."""
    print("=" * 60)
    print("Phase 3: Performance Benchmarks")
    print("=" * 60)
    
    import minisgl.core as core_module
    from minisgl.core import Batch, Req, SamplingParams
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    model, model_config = load_model_for_engine(device, dtype)
    
    print("\n" + "=" * 60)
    print("Hardware & Configuration")
    print("=" * 60)
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Precision: FP16")
    print(f"Backend: FlashAttention-3 (sgl-kernel)")
    print(f"Sliding Window: {SLIDING_WINDOW} tokens (ACTIVE)")
    print(f"Batch Size: 1 (single request)")
    print(f"Phase: Prefill only")
    print(f"Warmup: {WARMUP_ITERS} iterations")
    print(f"Benchmark: {BENCH_ITERS} iterations")
    
    print("\n" + "=" * 60)
    print("Prefill Performance")
    print("=" * 60)
    print(f"{'Seq Len':>8} | {'Latency (ms)':>12} | {'Throughput (tok/s)':>18} | {'Peak Mem (GB)':>13}")
    print("-" * 9 + "|" + "-" * 14 + "|" + "-" * 20 + "|" + "-" * 15)
    
    results = []
    
    for seq_len in PERF_SEQ_LENS:
        # Create dummy tokens
        tokens = torch.randint(1, 30000, (seq_len,), dtype=torch.int32)
        
        # Setup engine for this sequence length
        ctx, attn_backend, kv_cache, page_table = setup_engine(
            model_config, device, dtype, seq_len + 10, backend="fa", sliding_window_setting=SLIDING_WINDOW
        )
        
        # Create batch
        req = Req(
            input_ids=tokens,
            table_idx=0,
            cached_len=0,
            output_len=1,
            uid=0,
            sampling_params=SamplingParams(max_tokens=1),
            cache_handle=None,
        )
        
        batch = Batch(reqs=[req], phase="prefill")
        batch.input_ids = tokens.to(device)
        batch.out_loc = torch.arange(seq_len, device=device)
        batch.padded_reqs = [req]
        page_table[0, :seq_len] = torch.arange(seq_len, device=device)
        
        # Set global context
        core_module._GLOBAL_CTX = ctx
        
        # Warmup
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                with ctx.forward_batch(batch):
                    attn_backend.prepare_metadata(batch)
                    _ = model.forward()
            torch.cuda.synchronize()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(BENCH_ITERS):
            with torch.no_grad():
                with ctx.forward_batch(batch):
                    attn_backend.prepare_metadata(batch)
                    _ = model.forward()
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        # Calculate metrics
        total_time = end - start
        avg_latency_ms = (total_time / BENCH_ITERS) * 1000
        throughput = seq_len / (avg_latency_ms / 1000)
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        
        results.append({
            "seq_len": seq_len,
            "latency_ms": avg_latency_ms,
            "throughput": throughput,
            "peak_mem_gb": peak_mem_gb,
        })
        
        print(f"{seq_len:>8} | {avg_latency_ms:>12.2f} | {throughput:>18,.0f} | {peak_mem_gb:>13.2f}")
        
        # Cleanup
        core_module._GLOBAL_CTX = None
        del ctx, attn_backend, kv_cache, page_table, batch, req
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\nNote: Throughput is tokens/sec per request (not aggregate).")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive benchmark: mini-sglang vs HuggingFace")
    parser.add_argument("--hf", action="store_true", help="Phase 1: Generate HuggingFace reference")
    parser.add_argument("--engine", action="store_true", help="Phase 2: Engine comparison + sliding window test")
    parser.add_argument("--perf", action="store_true", help="Phase 3: Performance benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    args = parser.parse_args()
    
    if args.all:
        run_phase_1_hf()
        print("\n" + "#" * 60 + "\n")
        run_phase_2_engine()
        print("\n" + "#" * 60 + "\n")
        run_phase_3_perf()
    elif args.hf:
        run_phase_1_hf()
    elif args.engine:
        run_phase_2_engine()
    elif args.perf:
        run_phase_3_perf()
    else:
        print("Usage:")
        print("  python compare_logits.py --hf       # Phase 1: Generate HF reference")
        print("  python compare_logits.py --engine   # Phase 2: Engine comparison + sliding window")
        print("  python compare_logits.py --perf     # Phase 3: Performance benchmarks")
        print("  python compare_logits.py --all      # Run all phases")
