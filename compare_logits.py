"""
Compare logits between HuggingFace and mini-sglang implementations.

Usage:
  Step 1: python compare_logits.py --hf      # Generate HF reference
  Step 2: python compare_logits.py --engine  # Compare with engine
"""

import sys, os, torch, argparse, gc

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "python"))

MODEL_ID = "mistralai/Mistral-7B-v0.1"
TMP_FILE = "hf_reference.pt"

# Test prompts for comparison
TEST_PROMPTS = [
    "The capital of France is",
    "Machine learning is a type of",
    "def fibonacci(n):",
    "The quick brown fox",
]

def run_phase_1_hf():
    print("=" * 60)
    print("Phase 1: Collecting HuggingFace Reference Logits")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="cuda:0"
    )
    
    reference_data = []
    
    for text in TEST_PROMPTS:
        print(f"\nPrompt: '{text}'")
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        print(f"Tokens: {inputs.input_ids[0].tolist()}")
        
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :].float().cpu()
            reference_data.append({
                "logits": logits,
                "tokens": inputs.input_ids[0].cpu(),
                "text": text
            })
            
            probs = torch.softmax(logits[0], dim=-1)
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


def run_phase_2_engine():
    print("=" * 60)
    print("Phase 2: Running Mini-SGLang Model")
    print("=" * 60)
    
    # Patch ModelConfig.from_hf to handle Mistral's None head_dim
    from minisgl.models.config import ModelConfig, RotaryConfig
    
    @classmethod
    def patched_from_hf(cls, config):
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
        )
    
    ModelConfig.from_hf = patched_from_hf
    
    from minisgl.models.llama import LlamaForCausalLM
    from minisgl.models import ModelConfig
    from minisgl.layers import set_rope_device
    from minisgl.attention import create_attention_backend
    from minisgl.kvcache import create_kvcache
    from minisgl.core import Context, Batch, Req, SamplingParams, set_global_ctx
    from minisgl.distributed import set_tp_info
    from minisgl.utils import cached_load_hf_config, torch_dtype
    from transformers import AutoTokenizer
    
    if not os.path.exists(TMP_FILE):
        print(f"ERROR: {TMP_FILE} not found. Run with --hf first.")
        sys.exit(1)
    
    hf_data = torch.load(TMP_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    set_tp_info(rank=0, size=1)
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = torch.float16
    
    hf_config = cached_load_hf_config(MODEL_ID)
    model_config = ModelConfig.from_hf(hf_config)
    print(f"Model: {MODEL_ID}")
    print(f"Layers: {model_config.num_layers}, Heads: {model_config.num_qo_heads}, KV Heads: {model_config.num_kv_heads}")
    
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
    
    print("Setting up attention...")
    max_seq_len = 256
    max_batch_size = 4
    
    kv_cache = create_kvcache(
        model_config=model_config,
        num_pages=max_seq_len + 1,
        device=device,
        dtype=dtype,
    )
    
    page_table = torch.zeros((max_batch_size + 1, max_seq_len), dtype=torch.int32, device=device)
    
    attn_backend = create_attention_backend(
        "fi",
        model_config,
        kv_cache,
        page_table,
    )
    
    ctx = Context(page_size=1, attn_backend=attn_backend)
    set_global_ctx(ctx)
    
    print("\n" + "=" * 60)
    print("Comparing Logits")
    print("=" * 60)
    
    all_passed = True
    results = []
    
    for i, item in enumerate(hf_data):
        tokens = item["tokens"]
        text = item.get("text", "<unknown>")
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
        
        with torch.no_grad():
            with ctx.forward_batch(batch):
                attn_backend.prepare_metadata(batch)
                engine_logits = model.forward().float().cpu()
        
        hf_logits = item["logits"][0]
        eng_logits = engine_logits[-1]
        
        abs_diff = torch.abs(hf_logits - eng_logits)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        hf_probs = torch.softmax(hf_logits, dim=-1)
        eng_probs = torch.softmax(eng_logits, dim=-1)
        
        hf_top_ids = torch.topk(hf_probs, 5).indices
        eng_top_ids = torch.topk(eng_probs, 5).indices
        
        top1_match = hf_top_ids[0] == eng_top_ids[0]
        passed = max_diff < 0.1
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        results.append((text, max_diff, mean_diff, top1_match, passed))
        
        print(f"\n{status} Test {i+1}: '{text[:40]}...'" if len(text) > 40 else f"\n{status} Test {i+1}: '{text}'")
        print(f"   Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f} | Top-1 Match: {top1_match}")
        
        # Show top predictions for first test
        if i == 0:
            print("   Top-5: HF vs Engine")
            for j in range(5):
                hf_tok = tokenizer.decode([hf_top_ids[j]]).replace('\n', '\\n')
                eng_tok = tokenizer.decode([eng_top_ids[j]]).replace('\n', '\\n')
                print(f"     {j+1}. '{hf_tok}' ({hf_probs[hf_top_ids[j]]:.4f}) vs '{eng_tok}' ({eng_probs[eng_top_ids[j]]:.4f})")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed_count = sum(1 for r in results if r[4])
    total_count = len(results)
    avg_max_diff = sum(r[1] for r in results) / len(results)
    
    print(f"Passed: {passed_count}/{total_count}")
    print(f"Average Max Diff: {avg_max_diff:.6f}")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Engine matches HuggingFace!")
    else:
        print("\n⚠️  Some tests failed - check implementation")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare logits between HF and mini-sglang")
    parser.add_argument("--hf", action="store_true", help="Run Phase 1: HuggingFace reference")
    parser.add_argument("--engine", action="store_true", help="Run Phase 2: Engine comparison")
    args = parser.parse_args()
    
    if args.hf:
        run_phase_1_hf()
    elif args.engine:
        run_phase_2_engine()
    else:
        print("Usage:")
        print("  Step 1: python compare_logits.py --hf      # Generate HF reference")
        print("  Step 2: python compare_logits.py --engine  # Compare with engine")
