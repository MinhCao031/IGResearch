import os
import re
import json
import csv
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# =========================
# Config
# =========================
MODEL_ID    = "Helsinki-NLP/opus-mt-en-vi"
INPUT_CSV   = "rp1v301_iwslt15_envi.csv"   # CSV đã download sẵn — cột: source, reference
OUTPUT_JSON = "wp1v301_mt_translations.json"

MAX_NEW_TOKENS = 128
DO_SAMPLE      = False
NUM_BEAMS      = 4       # beam search — chất lượng tốt hơn greedy cho MT

# =========================
# Device
# =========================
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.float16 if device.type == "cuda" else torch.float32

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# Load model
# =========================
print(f"Loading model: {MODEL_ID}")
tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
model     = MarianMTModel.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
).to(device)
model.eval()

print(f"Model loaded. Parameters: {model.num_parameters()/1e6:.1f}M")

# =========================
# Load dataset từ CSV
# =========================
print(f"Reading CSV: {INPUT_CSV}")
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader   = csv.DictReader(f)
    test_set = list(reader)

print(f"Test set size: {len(test_set)} pairs")

# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def translate(source: str) -> str:
    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(device)

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=DO_SAMPLE,
                    early_stopping=True,
                )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=DO_SAMPLE,
                early_stopping=True,
            )

    hypothesis = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return clean_text(hypothesis)


# =========================
# Main loop
# =========================
print(f"\nBắt đầu dịch {len(test_set)} câu...")
results  = []
errors   = 0

for idx, row in enumerate(tqdm(test_set)):
    source    = clean_text(row.get("source", ""))
    reference = clean_text(row.get("reference", ""))

    if not source:
        errors += 1
        continue

    try:
        hypothesis = translate(source)
    except Exception as e:
        print(f"\n  [Error] ID {idx}: {e}")
        hypothesis = ""
        errors    += 1

    results.append({
        "id":         idx,
        "source":     source,      # câu tiếng Anh gốc
        "hypothesis": hypothesis,  # bản dịch của model
        "reference":  reference,   # bản dịch chuẩn từ dataset
    })

    if device.type == "cuda":
        torch.cuda.empty_cache()

# =========================
# Save
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nHOÀN THÀNH → {OUTPUT_JSON}")
print(f"Tổng:       {len(test_set)}")
print(f"Đã dịch:    {len(results)}")
print(f"Lỗi:        {errors}")
print(f"\nVí dụ output:")
for r in results[:3]:
    print(f"  [SRC] {r['source'][:80]}")
    print(f"  [HYP] {r['hypothesis'][:80]}")
    print(f"  [REF] {r['reference'][:80]}")
    print()