import json
import torch
import gc
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Config --------------------
DATA_PATH = "/content/drive/MyDrive/acereason11_100k.json"
OUTPUT_PATH = "best15k.json"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_SELECTED = 15000
MAX_LENGTH = 4096

# A100-OPTIMIZED SETTINGS (40GB VRAM)
BATCH_SIZE = 16

# Selection strategy
DIFFICULTY_PERCENTILE_LOW = 0.25
DIFFICULTY_PERCENTILE_HIGH = 0.75

# Checkpointing
CHECKPOINT_INTERVAL = 500
CHECKPOINT_FILE = Path(OUTPUT_PATH).with_suffix(".checkpoint.json")

# ------------------------------------------------

# Memory optimization
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_data(path):
    """Load JSON or JSONL dataset"""
    path = Path(path)
    if path.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict) and "data" in obj:
                return obj["data"]
            return obj

def compute_loss_single(model, tokenizer, example, failure_tracker):
    """
    Compute loss for single example with proper masking.
    Now tracks why examples fail.
    """
    try:
        # Extract prompt and output
        prompt = example.get("instruction", "")
        input_text = example.get("input", "")
        if input_text:
            prompt = f"{prompt}\n{input_text}"

        output = example.get("output") or example.get("solution") or example.get("response", "")

        if not output or not prompt:
            failure_tracker["missing_fields"] += 1
            return None

        # Tokenize prompt to get length
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)
        prompt_length = len(prompt_tokens["input_ids"])

        # Tokenize full text
        full_text = prompt + output
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True
        ).to(model.device)

        # Skip extremely long examples
        if inputs["input_ids"].shape[1] > MAX_LENGTH - 10:
            failure_tracker["too_long"] += 1
            return None

        # Create labels with masked prompt
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        # Compute loss
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()

        # Validate loss
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            failure_tracker["invalid_loss"] += 1
            return None

        return loss

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            failure_tracker["oom"] += 1
        else:
            failure_tracker["runtime_error"] += 1
        return None
    except Exception as e:
        failure_tracker["other_error"] += 1
        return None

def load_checkpoint():
    """Load checkpoint if exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return []

def save_checkpoint(scored_data):
    """Save checkpoint"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(scored_data, f)

def main():
    print(f" Loading model: {MODEL_NAME}")
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model optimized for A100
    print(" Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    # Clear cache after loading
    torch.cuda.empty_cache()
    gc.collect()

    print(f" Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f" Loading dataset from {DATA_PATH}")
    data = load_data(DATA_PATH)
    print(f"Total examples in file: {len(data)}")

    # Load checkpoint
    scored_data = load_checkpoint()
    start_idx = len(scored_data)

    if start_idx > 0:
        print(f"Resuming from index {start_idx}")

    # Track failures
    failure_tracker = {
        "missing_fields": 0,
        "too_long": 0,
        "invalid_loss": 0,
        "oom": 0,
        "runtime_error": 0,
        "other_error": 0
    }

    print(f"Scoring examples...")

    # Process examples
    for i in tqdm(range(start_idx, len(data)), initial=start_idx, total=len(data)):
        example = data[i]

        loss = compute_loss_single(model, tokenizer, example, failure_tracker)

        if loss is not None:
            scored_data.append({
                "index": i,
                "loss": loss,
                "example": example
            })

        # Periodic cache clearing
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(scored_data)
            # Print progress with failure stats
            if len(scored_data) > 0:
                current_vram = torch.cuda.memory_allocated() / 1e9
                success_rate = len(scored_data) / (i + 1) * 100
                print(f"\n Checkpoint: {len(scored_data)}/{i+1} scored ({success_rate:.1f}% success) | VRAM: {current_vram:.2f} GB")

    save_checkpoint(scored_data)
    print(f"\n Scoring complete!")
    print(f"   Successfully scored: {len(scored_data)}")
    print(f"   Failed: {len(data) - len(scored_data)}")

    # Print failure breakdown
    print(f"\n Failure Breakdown:")
    total_failures = sum(failure_tracker.values())
    for reason, count in failure_tracker.items():
        if count > 0:
            percentage = (count / total_failures * 100) if total_failures > 0 else 0
            print(f"   {reason}: {count} ({percentage:.1f}%)")

    if len(scored_data) == 0:
        print("\n No valid examples scored. Check data format.")
        return

    # Sort and select
    print("\n Sorting by difficulty...")
    scored_data.sort(key=lambda x: x["loss"])

    n_total = len(scored_data)
    low_idx = int(n_total * DIFFICULTY_PERCENTILE_LOW)
    high_idx = int(n_total * DIFFICULTY_PERCENTILE_HIGH)

    intermediate_range = scored_data[low_idx:high_idx]

    print(f"\n Difficulty Distribution:")
    print(f"   Total valid: {n_total}")
    print(f"   Too easy (bottom {DIFFICULTY_PERCENTILE_LOW:.0%}): {low_idx} examples → SKIPPED")
    print(f"   Optimal range ({DIFFICULTY_PERCENTILE_LOW:.0%}-{DIFFICULTY_PERCENTILE_HIGH:.0%}): {len(intermediate_range)} examples")
    print(f"   Too hard (top {1-DIFFICULTY_PERCENTILE_HIGH:.0%}): {n_total - high_idx} examples → SKIPPED")

    if len(intermediate_range) <= NUM_SELECTED:
        selected_data = intermediate_range
        print(f"\n  Selected all {len(selected_data)} from optimal range")
    else:
        step = len(intermediate_range) / NUM_SELECTED
        selected_indices = [int(i * step) for i in range(NUM_SELECTED)]
        selected_data = [intermediate_range[idx] for idx in selected_indices]
        print(f"\n  Selected {len(selected_data)} from optimal range")

    final_examples = [item["example"] for item in selected_data]

    # Statistics
    avg_loss = sum(item["loss"] for item in selected_data) / len(selected_data)
    min_loss = min(item["loss"] for item in selected_data)
    max_loss = max(item["loss"] for item in selected_data)

    print(f"\n Selected Examples Statistics:")
    print(f"   Count: {len(final_examples)}")
    print(f"   Loss range: {min_loss:.4f} - {max_loss:.4f}")
    print(f"   Average loss: {avg_loss:.4f}")

    # Save results
    print(f"\n Saving to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_examples, f, indent=2, ensure_ascii=False)

    # Save metadata
    metadata = {
        "total_scored": n_total,
        "total_in_file": len(data),
        "selected_count": len(final_examples),
        "difficulty_percentile_low": DIFFICULTY_PERCENTILE_LOW,
        "difficulty_percentile_high": DIFFICULTY_PERCENTILE_HIGH,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "avg_loss": avg_loss,
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "failure_breakdown": failure_tracker
    }

    metadata_path = Path(OUTPUT_PATH).with_suffix(".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f" Metadata saved to {metadata_path}")

    # Cleanup
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("🧹 Checkpoint file removed")

    print("\n Data selection finished successfully!")
    print(f" Final VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB / 40 GB")

if __name__ == "__main__":
    main()
