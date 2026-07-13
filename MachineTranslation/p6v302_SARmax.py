"""
P6 (SAR max) — Phiên bản thay thế dùng MAX thay vì SUM.

SAR thường:
  SAR = sum(pos attribution của content words) / sum(tất cả pos attribution)
  → Đo tỷ lệ tổng thể, dễ bị pha loãng bởi nhiều token nhỏ.

SAR max:
  SAR_max = max(pos attribution của content words) / max(tất cả pos attribution)
  → Đo xem token quan trọng nhất của câu nguồn có đóng góp lớn nhất không.
  → Ít bị ảnh hưởng bởi số lượng token, nhạy hơn với token đơn lẻ nổi bật.

Nhãn hallucination lấy từ P5 (classify_label: correct / hallucinated).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy import stats as sp_stats

# =========================
# Config
# =========================
IG_FILE    = "wp4v301_mt_ig_nllb.json"     # có source_tokens + attribution_scores
LABEL_FILE = "wp5v301_mt_classify.json"  # có classify_label + chrf_score
OUTPUT_JSON = "wp6v302_mt_sar_max.json"
OUTPUT_PNG  = "wp6v302_mt_sar_max.png"

# =========================
# Tính SAR max
# =========================
def calculate_sar(attribution_scores: list[dict], source_tokens: list[dict]) -> float | None:
    """
    SAR_max = max(positive attribution của content words)
            / max(tất cả positive attribution trong câu nguồn)

    Ý nghĩa: token quan trọng nhất trong câu nguồn (content word có attribution cao nhất)
    chiếm bao nhiêu phần trăm so với token có attribution cao nhất toàn câu.

    SAR_max cao → token quan trọng nhất của câu nguồn đang dẫn dắt bản dịch → grounded.
    SAR_max thấp → token có attribution cao nhất là stop word / padding → không grounded.

    attribution_scores: list[{t: str, s: float}]  — 1 entry / source token
    source_tokens:      list[{token, char_start, char_end}]  — content words từ P2
    """
    if not attribution_scores or not source_tokens:
        return None

    src_token_strs = {
        t["token"].lower().strip().lstrip("▁")
        for t in source_tokens
    }

    max_total  = 0.0   # max attribution dương trong toàn bộ câu nguồn
    max_source = 0.0   # max attribution dương trong nhóm content words

    for entry in attribution_scores:
        s = entry["s"]
        if s <= 0:
            continue

        # Strip NLLB subword prefix ▁ trước khi match
        t = entry["t"].lower().strip().lstrip("▁")

        if s > max_total:
            max_total = s

        # Exact match với content words
        if t in src_token_strs:
            if s > max_source:
                max_source = s

    if max_total == 0:
        return None

    return max_source / max_total


# =========================
# Load data
# =========================
print("Đang tải dữ liệu...")

with open(IG_FILE, "r", encoding="utf-8") as f:
    ig_data = json.load(f)

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    label_data = json.load(f)

# Build label map từ P5
label_map: dict[int, dict] = {}
for item in label_data:
    label  = item.get("classify_label", "")
    status = item.get("classify_status", "")
    if label not in ("correct", "hallucinated"):
        continue
    label_map[int(item["id"])] = {
        "label":      label,
        "chrf_score": item.get("chrf_score", 0.0),
        "bleu_score": item.get("bleu_score", 0.0),
    }

print(f"Labels loaded: {len(label_map)} items")

# =========================
# Tính SAR cho từng mẫu
# =========================
print("Đang tính SAR...")

data_rows = []
skipped   = 0
sar_results = []

for item in ig_data:
    item_id = int(item.get("id", -1))

    if item_id not in label_map:
        skipped += 1
        continue

    label_info = label_map[item_id]
    label      = label_info["label"]
    label_code = 1 if label == "hallucinated" else 0
    class_name = "Hallucination" if label_code == 1 else "Truth"

    attribution_scores = item.get("attribution_scores", [])
    source_tokens      = item.get("source_tokens", [])

    sar = calculate_sar(attribution_scores, source_tokens)

    if sar is None:
        skipped += 1
        continue

    data_rows.append({
        "id":         item_id,
        "SAR":        sar,       # SAR_max value — giữ key "SAR" để tương thích downstream
        "label_code": label_code,
        "Class":      class_name,
        "chrf_score": label_info["chrf_score"],
    })

    sar_results.append({
        "id":             item_id,
        "source":         item.get("source", ""),
        "hypothesis":     item.get("hypothesis", ""),
        "reference":      item.get("reference", ""),
        "SAR_max":        sar,
        "classify_label": label,
        "chrf_score":     label_info["chrf_score"],
    })

df = pd.DataFrame(data_rows)
print(f"SAR computed: {len(df)} items  (skipped={skipped})")

# =========================
# Vẽ biểu đồ
# =========================
print("\nĐang tạo biểu đồ...")
sns.set_theme(style="whitegrid", context="talk")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
palette = {
    "Truth":         "#2ca02c",
    "Hallucination": "#d62728",
}

# Biểu đồ 1: KDE phân phối SAR
sns.kdeplot(
    data=df, x="SAR", hue="Class", fill=True, common_norm=False,
    palette=palette, alpha=0.4, linewidth=2.5, ax=axes[0], cut=0,
)
axes[0].set_title("Phân phối SAR_max — Dịch máy En→Vi", fontweight="bold")
axes[0].set_xlabel("Điểm SAR_max (0.0 → 1.0)")
axes[0].set_ylabel("Mật độ (Density)")

# Biểu đồ 2: ROC Curve
y_true  = df["label_code"]
fpr, tpr, thresholds = roc_curve(y_true, -df["SAR"])   # SAR thấp → hallucination
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color="blue", lw=3,
             label=f"SAR_max (AUC = {roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
axes[1].set_xlim([-0.02, 1.02])
axes[1].set_ylim([-0.02, 1.05])
axes[1].set_xlabel("Tỷ lệ Nhận diện nhầm (FPR)")
axes[1].set_ylabel("Tỷ lệ Nhận diện đúng (TPR)")
axes[1].set_title("Đường cong ROC — SAR vs Hallucination", fontweight="bold")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"  → Đã xuất: {OUTPUT_PNG}")

# =========================
# Thống kê mô tả
# =========================
grp_c = df.loc[df["label_code"] == 0, "SAR"]
grp_h = df.loc[df["label_code"] == 1, "SAR"]

u_stat, p_value = sp_stats.mannwhitneyu(grp_c, grp_h, alternative="two-sided")
r_rb = 1 - 2 * u_stat / (len(grp_c) * len(grp_h))

print(f"\n{'='*55}")
print(f" THỐNG KÊ MÔ TẢ — SAR_max (MT En→Vi)")
print(f"{'='*55}")
print(f"\n  Truth        n={len(grp_c):4d}  mean={grp_c.mean():.4f}  std={grp_c.std():.4f}")
print(f"  Hallucinated n={len(grp_h):4d}  mean={grp_h.mean():.4f}  std={grp_h.std():.4f}")
print(f"\n  Mann-Whitney U = {u_stat:.0f}")
print(f"  p-value        = {p_value:.4f}  {'✓ significant (p<0.05)' if p_value < 0.05 else '✗ not significant'}")
print(f"  Rank-biserial r= {r_rb:+.4f}")
print(f"\n  AUC SAR        = {roc_auc:.4f}")

# Tìm ngưỡng tốt nhất trên ROC
best_idx       = np.argmax(tpr - fpr)
best_threshold = -thresholds[best_idx]
best_tpr       = tpr[best_idx]
best_fpr       = fpr[best_idx]
print(f"\n  Best threshold (Youden's J):")
print(f"    SAR < {best_threshold:.4f} → hallucinated")
print(f"    TPR = {best_tpr:.3f}, FPR = {best_fpr:.3f}")
print(f"{'='*55}")

# =========================
# Scatter plot thêm: SAR vs chrF
# =========================
fig2, ax2 = plt.subplots(figsize=(8, 6))
scatter_palette = {"Truth": "#2ca02c", "Hallucination": "#d62728"}
for cls, grp in df.groupby("Class"):
    ax2.scatter(
        grp["SAR"], grp["chrf_score"],
        label=cls, alpha=0.4, s=20,
        color=scatter_palette[cls],
    )
ax2.set_xlabel("SAR Score")
ax2.set_ylabel("chrF Score")
ax2.set_title("SAR_max vs chrF — Tương quan với chất lượng dịch", fontweight="bold")
ax2.legend()
ax2.axhline(y=40, color="gray", linestyle="--", alpha=0.5, label="chrF threshold")
plt.tight_layout()
scatter_path = OUTPUT_PNG.replace(".png", "_scatter.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
print(f"  → Scatter plot: {scatter_path}")

# =========================
# Save results
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(sar_results, f, ensure_ascii=False, indent=2)

print(f"\nHOÀN THÀNH → {OUTPUT_JSON}")