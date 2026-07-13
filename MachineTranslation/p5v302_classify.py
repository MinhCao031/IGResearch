"""
P5 (BERTScore) — Gán nhãn hallucination dựa trên semantic similarity.

Thay chrF (character n-gram) bằng BERTScore (contextual embedding similarity).
BERTScore nắm bắt ngữ nghĩa tốt hơn:
  - Hai câu cùng nghĩa nhưng khác từ vẫn có score cao
  - Dùng xlm-roberta-base — hỗ trợ tiếng Việt tốt
  - Đo F1 giữa token embeddings của hypothesis và reference

Nhãn:
  - "correct"     : BERTScore F1 >= THRESHOLD
  - "hallucinated": BERTScore F1 <  THRESHOLD

THRESHOLD mặc định = 0.85 (thang 0.0–1.0).
Nên calibrate trên subset nhỏ trước khi chạy toàn bộ.
"""

import json
import torch
from bert_score import score as bert_score
from sacrebleu.metrics import CHRF
from tqdm import tqdm

# =========================
# Config
# =========================
INPUT_JSON  = "wp4v301_mt_ig_nllb.json"
OUTPUT_JSON = "wp5v301_mt_classify_bertscore.json"

THRESHOLD   = 0.85    # BERTScore F1 ∈ [0, 1] — điều chỉnh sau khi calibrate
BERT_MODEL  = "FacebookAI/xlm-roberta-base"   # multilingual, hỗ trợ tiếng Việt tốt
BATCH_SIZE  = 64      # tăng nếu RAM đủ, giảm nếu OOM
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# chrF giữ lại để tham khảo / so sánh
chrf_metric = CHRF()

# =========================
# Tính BERTScore batch — hiệu quả hơn gọi từng câu
# =========================
def compute_bertscore_batch(hypotheses: list[str], references: list[str]) -> list[float]:
    """
    Tính BERTScore F1 cho một batch cặp (hypothesis, reference).
    Trả về list float ∈ [0, 1].
    """
    P, R, F1 = bert_score(
        hypotheses,
        references,
        model_type=BERT_MODEL,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        lang="vi",          # gợi ý ngôn ngữ đích
        verbose=False,
    )
    return F1.tolist()


def score_item_single(hypothesis: str, reference: str, bert_f1: float) -> dict:
    """
    Gán nhãn dựa trên BERTScore F1 đã tính.
    Tính thêm chrF để tham khảo.
    """
    if not hypothesis or not reference:
        return {
            "bertscore_f1": 0.0,
            "chrf_score":   0.0,
            "label":        "hallucinated",
            "status":       "MISSING_TEXT",
        }

    chrf_score = chrf_metric.sentence_score(hypothesis, [reference]).score
    label      = "correct" if bert_f1 >= THRESHOLD else "hallucinated"

    return {
        "bertscore_f1": round(bert_f1, 4),
        "chrf_score":   round(chrf_score, 4),
        "label":        label,
        "status":       "OK",
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