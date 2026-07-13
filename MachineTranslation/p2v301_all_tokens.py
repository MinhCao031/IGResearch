"""
P2 — Trích xuất TẤT CẢ token từ source (tiếng Anh) và hypothesis (tiếng Việt).

Không lọc theo POS tag, không bỏ auxiliary/stopword — giữ lại toàn bộ từ:
  - Source (EN): nltk word_tokenize (tokenize thô, không POS tag)
  - Target (VI): underthesea word_tokenize (tokenize thô, không POS tag)

Mỗi token output có vị trí ký tự (char_start, char_end) trong câu gốc
để P4 sau này không bị nhầm khi cùng từ xuất hiện nhiều lần.
"""

import json
import re
import sys
from tqdm import tqdm

# =========================
# Thư viện NLP
# =========================
import nltk
from nltk import word_tokenize

# Download nếu chưa có
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

try:
    from underthesea import word_tokenize as vi_word_tokenize
    UNDERTHESEA_OK = True
except ImportError:
    print("[WARN] underthesea chưa được cài. Chạy: pip install underthesea")
    print("       Sẽ fallback về tokenize đơn giản cho tiếng Việt.")
    UNDERTHESEA_OK = False

# =========================
# Config
# =========================
INPUT_JSON  = "wp1v301_mt_translations_nllb.json"   # output từ script dịch NLLB
OUTPUT_JSON = "wp2v301_mt_tokens_nllb.json"


# =========================
# Helpers chung
# =========================
def find_char_position(token: str, text: str, start_from: int = 0) -> tuple[int, int] | None:
    """
    Tìm vị trí ký tự đầu tiên của token trong text, bắt đầu từ start_from.
    Case-insensitive. Trả về (char_start, char_end) hoặc None.
    """
    idx = text.lower().find(token.lower(), start_from)
    if idx == -1:
        return None
    return (idx, idx + len(token))


def tokens_with_positions(tokens: list[str], text: str) -> list[dict]:
    """
    Với mỗi token, tìm vị trí trong text theo thứ tự xuất hiện.
    Tránh nhầm khi cùng từ xuất hiện nhiều lần bằng cách tìm từ vị trí
    lần xuất hiện trước đó.
    """
    result    = []
    last_end  = 0

    for token in tokens:
        pos = find_char_position(token, text, last_end)
        if pos is None:
            # Thử từ đầu (trường hợp token xuất hiện trước vị trí hiện tại)
            pos = find_char_position(token, text, 0)

        if pos is not None:
            result.append({
                "token":      token,
                "char_start": pos[0],
                "char_end":   pos[1],
            })
            last_end = pos[1]
        # Nếu không tìm được vị trí → drop (không nên xảy ra)

    return result


# =========================
# Tiếng Anh — source tokens (TẤT CẢ, không lọc)
# =========================
def extract_en_all_words(text: str) -> list[str]:
    """
    Tokenize toàn bộ câu tiếng Anh, giữ lại mọi token — kể cả
    auxiliary verb, stopword, dấu câu. Không dùng POS tagging để lọc.
    """
    return word_tokenize(text)


# =========================
# Tiếng Việt — target tokens (TẤT CẢ, không lọc)
# =========================
def extract_vi_all_words(text: str) -> list[str]:
    """
    Tokenize toàn bộ câu tiếng Việt, giữ lại mọi từ — kể cả
    stopword, dấu câu. Dùng underthesea word_tokenize (word segmentation,
    không POS tag) nếu có, fallback về split() đơn giản.
    """
    if UNDERTHESEA_OK:
        try:
            return vi_word_tokenize(text)
        except Exception as e:
            print(f"  [underthesea error] {e} — fallback")

    # Fallback: tokenize đơn giản theo khoảng trắng, không lọc gì cả
    return text.split()


# =========================
# Main
# =========================
print(f"Reading: {INPUT_JSON}")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total: {len(data)} items")

results  = []
warnings = 0

for item in tqdm(data):
    source     = item["source"]
    hypothesis = item["hypothesis"]
    reference  = item.get("reference", "")

    # Trích xuất TẤT CẢ token, không lọc
    src_words = extract_en_all_words(source)
    tgt_words = extract_vi_all_words(hypothesis)

    # Gán vị trí ký tự
    source_tokens = tokens_with_positions(src_words, source)
    target_tokens = tokens_with_positions(tgt_words, hypothesis)

    # Kiểm tra
    if not source_tokens:
        print(f"\n  [WARN] ID {item['id']}: không có source token — {source[:60]}")
        warnings += 1
    if not target_tokens:
        print(f"\n  [WARN] ID {item['id']}: không có target token — {hypothesis[:60]}")
        warnings += 1

    results.append({
        "id":            item["id"],
        "source":        source,
        "hypothesis":    hypothesis,
        "reference":     reference,
        "source_tokens": source_tokens,   # list[{token, char_start, char_end}]
        "target_tokens": target_tokens,   # list[{token, char_start, char_end}]
    })

# =========================
# Save
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Thống kê
src_lens = [len(r["source_tokens"]) for r in results]
tgt_lens = [len(r["target_tokens"]) for r in results]

print(f"\nHOÀN THÀNH → {OUTPUT_JSON}")
print(f"Tổng:              {len(results)}")
print(f"Warnings:          {warnings}")
print(f"Source tokens/câu: min={min(src_lens)} avg={sum(src_lens)/len(src_lens):.1f} max={max(src_lens)}")
print(f"Target tokens/câu: min={min(tgt_lens)} avg={sum(tgt_lens)/len(tgt_lens):.1f} max={max(tgt_lens)}")

# Ví dụ
print(f"\nVí dụ:")
for r in results[:2]:
    print(f"  [SRC] {r['source'][:70]}")
    print(f"  [SRC tokens] {[t['token'] for t in r['source_tokens']]}")
    print(f"  [HYP] {r['hypothesis'][:70]}")
    print(f"  [TGT tokens] {[t['token'] for t in r['target_tokens']]}")
    print()