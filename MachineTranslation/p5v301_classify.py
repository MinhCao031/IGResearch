"""
P5 — Gán nhãn hallucination cho bản dịch dựa trên chrF score.

So sánh hypothesis với reference bằng chrF (character n-gram F-score).
chrF phù hợp hơn BLEU cho tiếng Việt vì:
  - Tiếng Việt có nhiều từ ghép và dấu thanh
  - chrF đo ở cấp độ ký tự → ít bị phạt vì cách diễn đạt khác nhau hơn BLEU

Nhãn:
  - "correct"     : chrF >= THRESHOLD
  - "hallucinated": chrF <  THRESHOLD

THRESHOLD mặc định = 40.0 (thang 0–100).
Nên thử calibrate trên subset nhỏ trước khi chạy toàn bộ.
"""

import json
from sacrebleu.metrics import CHRF, BLEU
from tqdm import tqdm

# =========================
# Config
# =========================
INPUT_JSON  = "wp4v301_mt_ig.json"    # output của P4
OUTPUT_JSON = "wp5v301_mt_classify.json"

THRESHOLD   = 40.0    # chrF score — điều chỉnh sau khi calibrate

# =========================
# Load metrics
# =========================
chrf_metric = CHRF()
bleu_metric = BLEU(effective_order=True)   # sentence-level BLEU

# =========================
# Helpers
# =========================
def score_item(hypothesis: str, reference: str) -> dict:
    """
    Tính chrF và BLEU cho một cặp hypothesis/reference.
    Trả về dict gồm scores và nhãn.
    """
    if not hypothesis or not reference:
        return {
            "chrf_score":  0.0,
            "bleu_score":  0.0,
            "label":       "hallucinated",
            "status":      "MISSING_TEXT",
        }

    chrf_score = chrf_metric.sentence_score(
        hypothesis, [reference]
    ).score

    bleu_score = bleu_metric.sentence_score(
        hypothesis, [reference]
    ).score

    label  = "correct" if chrf_score >= THRESHOLD else "hallucinated"

    return {
        "chrf_score": round(chrf_score, 4),
        "bleu_score": round(bleu_score, 4),
        "label":      label,
        "status":     "OK",
    }


# =========================
# Main
# =========================
print(f"Reading: {INPUT_JSON}")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total: {len(data)} items")
print(f"Threshold: chrF >= {THRESHOLD} → correct\n")

results      = []
failed_items = []
status_counts = {"correct": 0, "hallucinated": 0, "MISSING_TEXT": 0}

for item in tqdm(data):
    hypothesis = item.get("hypothesis", "")
    reference  = item.get("reference",  "")

    scored = score_item(hypothesis, reference)

    item["chrf_score"]      = scored["chrf_score"]
    item["bleu_score"]      = scored["bleu_score"]
    item["classify_label"]  = scored["label"]
    item["classify_status"] = scored["status"]

    results.append(item)

    status_counts[scored["label"]] = status_counts.get(scored["label"], 0) + 1
    if scored["status"] != "OK":
        status_counts["MISSING_TEXT"] += 1
        failed_items.append(item)

# =========================
# Save
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

failed_output = OUTPUT_JSON.replace(".json", "_failed.json")
with open(failed_output, "w", encoding="utf-8") as f:
    json.dump(failed_items, f, indent=2, ensure_ascii=False)

# =========================
# Summary
# =========================
chrf_scores = [r["chrf_score"] for r in results if r["classify_status"] == "OK"]
n_correct      = status_counts.get("correct", 0)
n_hallucinated = status_counts.get("hallucinated", 0)

print(f"\nHOÀN THÀNH → {OUTPUT_JSON}")
print(f"Tổng:              {len(data)}")
print(f"Đã xử lý:          {len(results)}")
print(f"  correct:         {n_correct}  ({n_correct/len(results)*100:.1f}%)")
print(f"  hallucinated:    {n_hallucinated}  ({n_hallucinated/len(results)*100:.1f}%)")
print(f"  missing text:    {status_counts['MISSING_TEXT']}")
print(f"\nchrF stats:")
print(f"  min:  {min(chrf_scores):.2f}")
print(f"  max:  {max(chrf_scores):.2f}")
print(f"  avg:  {sum(chrf_scores)/len(chrf_scores):.2f}")
print(f"  median: {sorted(chrf_scores)[len(chrf_scores)//2]:.2f}")
print(f"\nThreshold dùng: {THRESHOLD}")
print(f"Failed saved to: {failed_output}")

# =========================
# Gợi ý calibrate threshold
# =========================
print(f"\n--- Phân phối chrF để calibrate threshold ---")
buckets = [(0,20), (20,30), (30,40), (40,50), (50,60), (60,70), (70,100)]
for lo, hi in buckets:
    count = sum(1 for s in chrf_scores if lo <= s < hi)
    bar   = "█" * (count * 40 // len(chrf_scores))
    print(f"  [{lo:3d}–{hi:3d}): {count:4d}  {bar}")