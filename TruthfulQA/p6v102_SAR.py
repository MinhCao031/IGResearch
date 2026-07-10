import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ==========================================
# 0. CẤU HÌNH ĐƯỜNG DẪN FILE
# ==========================================
LABEL_FILE = "wp5v100_classify.json"   # .json hoặc .csv đều được (xem mục 2)
IG_FILE    = "wp4v101_qa_ig.json"

# ==========================================
# 1. HÀM TÍNH TOÁN SAR (đã sửa 4 lỗi)
# ==========================================

def calculate_sar(ig_list: list, source_tokens: list) -> float | None:
    """
    SAR (Source Attribution Ratio) — tỷ lệ ảnh hưởng dương từ source tokens
    trên tổng ảnh hưởng dương của toàn bộ câu hỏi.

    SAR cao → câu trả lời được dẫn dắt bởi source tokens → grounded
    SAR thấp → câu trả lời không bắt nguồn từ source → khả năng hallucination

    FIX 1: Weighted average across ALL target tokens (không chỉ ig_list[0]).
            Weight = total positive attribution của từng target (target có nhiều
            ảnh hưởng hơn sẽ đóng góp nhiều hơn vào SAR tổng).
    """
    if not ig_list:
        return None

    src_lower = [s.lower().strip() for s in source_tokens if s.strip()]

    total_weight   = 0.0
    weighted_sar   = 0.0

    for entry in ig_list:
        attrs = entry.get("attribution_scores", [])
        if not attrs:
            continue

        # FIX 2: Tách positive và negative attribution của source tokens
        pos_total   = sum(a["s"] for a in attrs if a["s"] > 0)
        if pos_total == 0:
            continue                     # target token này không có ảnh hưởng dương nào

        src_pos_anchor = sum(
            a["s"] for a in attrs
            if a["s"] > 0
            and len(a["t"].strip()) > 1
            and any(st in a["t"].lower() or a["t"].lower() in st for st in src_lower)
        )

        sar_i = src_pos_anchor / pos_total   # SAR của target token này ∈ [0, 1]

        # Weight theo tổng positive attribution (target có signal mạnh hơn = quan trọng hơn)
        weighted_sar += sar_i * pos_total
        total_weight += pos_total

    if total_weight == 0:
        return None

    return weighted_sar / total_weight



# ==========================================
# 2. ĐỌC VÀ KẾT NỐI DỮ LIỆU
# ==========================================
print("Đang tải dữ liệu và tính toán SAR...")

# FIX 3: Hỗ trợ thực sự cả .csv lẫn .json; skip nhãn 'invalid'
label_map: dict[int, int] = {}

if LABEL_FILE.endswith(".csv"):
    lbl_df = pd.read_csv(LABEL_FILE)
    for _, row in lbl_df.iterrows():
        lbl_str = str(row.get("classify_label", "")).strip().lower()
        if lbl_str not in ("correct", "incorrect"):   # FIX 3: skip invalid
            continue
        label_map[int(row["id"])] = 1 if lbl_str == "incorrect" else 0
else:
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        lbl_data = json.load(f)
    for item in lbl_data:
        lbl_str = str(item.get("classify_label", "")).strip().lower()
        if lbl_str not in ("correct", "incorrect"):   # FIX 3: skip invalid
            continue
        label_map[int(item["id"])] = 1 if lbl_str == "incorrect" else 0

print(f"  Labels loaded: {len(label_map)} items "
      f"(correct={sum(1 for v in label_map.values() if v==0)}, "
      f"incorrect={sum(1 for v in label_map.values() if v==1)})")

# Đọc IG và tính SAR
with open(IG_FILE, "r", encoding="utf-8") as f:
    ig_dataset = json.load(f)

data_rows = []
skipped   = 0

for item in ig_dataset:
    item_id = int(item.get("id", -1))
    if item_id not in label_map:
        skipped += 1
        continue

    label_code = label_map[item_id]
    class_name = "Hallucination (INCORRECT)" if label_code == 1 else "Truth (CORRECT)"

    ig_list       = item.get("ig", [])
    source_tokens = item.get("source_tokens", [])

    sar = calculate_sar(ig_list, source_tokens)

    if sar is None:
        skipped += 1
        continue

    data_rows.append({
        "id":         item_id,
        "SAR":        sar,
        "label_code": label_code,
        "Class":      class_name,
    })

df = pd.DataFrame(data_rows)
print(f"  SAR computed: {len(df)} items  (skipped={skipped})")

# ==========================================
# 3. VẼ BIỂU ĐỒ BÁO CÁO SAR
# ==========================================
print("\nĐang tạo biểu đồ SAR Report...")
sns.set_theme(style="whitegrid", context="talk")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
palette = {
    "Truth (CORRECT)":           "#2ca02c",
    "Hallucination (INCORRECT)": "#d62728",
}

# --- Biểu đồ 1: KDE phân phối SAR ---
sns.kdeplot(
    data=df, x="SAR", hue="Class", fill=True, common_norm=False,
    palette=palette, alpha=0.4, linewidth=2.5, ax=axes[0], cut=0
)
axes[0].set_title("Phân phối điểm SAR (Source Attribution Ratio)", fontweight="bold")
axes[0].set_xlabel("Điểm SAR (0.0 → 1.0)")
axes[0].set_ylabel("Mật độ (Density)")

# --- Biểu đồ 2: ROC Curve ---
y_true = df["label_code"]
fpr, tpr, _ = roc_curve(y_true, -df["SAR"])   # SAR thấp → hallucination
roc_auc     = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color="blue", lw=3, label=f"SAR Method (AUC = {roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
axes[1].set_xlim([-0.02, 1.02])
axes[1].set_ylim([-0.02, 1.05])
axes[1].set_xlabel("Tỷ lệ Nhận diện nhầm (FPR)")
axes[1].set_ylabel("Tỷ lệ Nhận diện đúng (TPR)")
axes[1].set_title("Đường cong ROC: Khả năng phân loại của SAR", fontweight="bold")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("wp6v102_SAR.png", dpi=300, bbox_inches="tight")
print("  -> Đã xuất: wp6v102_SAR.png")

# ==========================================
# 4. THỐNG KÊ MÔ TẢ
# ==========================================
from scipy import stats as sp_stats

print("\n" + "=" * 55)
print(" THỐNG KÊ MÔ TẢ — SAR")
print("=" * 55)

grp_c = df.loc[df["label_code"] == 0, "SAR"]
grp_i = df.loc[df["label_code"] == 1, "SAR"]
u, p  = sp_stats.mannwhitneyu(grp_c, grp_i, alternative="two-sided")
r_rb  = 1 - 2 * u / (len(grp_c) * len(grp_i))

print(f"\n[SAR]")
print(f"  correct   n={len(grp_c)}  mean={grp_c.mean():.4f}  std={grp_c.std():.4f}")
print(f"  incorrect n={len(grp_i)}  mean={grp_i.mean():.4f}  std={grp_i.std():.4f}")
print(f"  Mann-Whitney p={p:.4f}  rank-biserial r={r_rb:+.4f}")
print(f"\n{'AUC SAR':15s}: {roc_auc:.4f}")
print("=" * 55)
print("\nHOÀN THÀNH")