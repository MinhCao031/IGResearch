"""
P2 — Trích xuất content words từ source (tiếng Anh) và hypothesis (tiếng Việt).

Không dùng LLM — dùng POS tagging:
  - Source (EN): nltk + averaged_perceptron_tagger
  - Target (VI): underthesea

Content words được giữ lại:
  - Noun (danh từ)
  - Verb chính (không phải auxiliary)
  - Adjective (tính từ)
  - Number / Named entity

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
from nltk import pos_tag, word_tokenize

# Download nếu chưa có
for resource in ["averaged_perceptron_tagger", "punkt", "punkt_tab",
                 "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

try:
    from underthesea import pos_tag as vi_pos_tag
    UNDERTHESEA_OK = True
except ImportError:
    print("[WARN] underthesea chưa được cài. Chạy: pip install underthesea")
    print("       Sẽ fallback về tokenize đơn giản cho tiếng Việt.")
    UNDERTHESEA_OK = False

# =========================
# Config
# =========================
INPUT_JSON  = "wp1v301_mt_translations.json"
OUTPUT_JSON = "wp2v301_mt_tokens.json"

# POS tags tiếng Anh cần giữ lại
EN_KEEP_TAGS = {
    "NN", "NNS", "NNP", "NNPS",          # Noun
    "VB", "VBD", "VBG", "VBN",            # Verb
    "VBP", "VBZ",
    "JJ", "JJR", "JJS",                   # Adjective
    "CD",                                  # Number
    "FW",                                  # Foreign word
}

# Auxiliary verbs tiếng Anh — loại bỏ dù là VB
EN_AUXILIARIES = {
    "be", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should",
    "may", "might", "must", "can", "could",
}

# POS tags tiếng Việt cần giữ lại (underthesea convention)
VI_KEEP_TAGS = {
    "N", "Np",   # Noun, Proper noun
    "V",         # Verb
    "A",         # Adjective
    "M",         # Number
    "Nu",        # Unit
}


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
# Tiếng Anh — source tokens
# =========================
def extract_en_content_words(text: str) -> list[str]:
    """
    Trích xuất content words từ câu tiếng Anh bằng NLTK POS tagging.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    result = []
    for word, tag in tagged:
        if tag not in EN_KEEP_TAGS:
            continue
        if word.lower() in EN_AUXILIARIES:
            continue
        if len(word) <= 1:
            continue
        if not re.search(r"[a-zA-Z0-9]", word):
            continue
        result.append(word)

    return result


# =========================
# Tiếng Việt — target tokens
# =========================
def extract_vi_content_words(text: str) -> list[str]:
    """
    Trích xuất content words từ câu tiếng Việt.
    Dùng underthesea nếu có, fallback về tokenize đơn giản.
    """
    if UNDERTHESEA_OK:
        try:
            tagged = vi_pos_tag(text)
            # underthesea trả về list of (word, tag)
            result = []
            for item in tagged:
                word = item[0] if isinstance(item, (list, tuple)) else item
                tag  = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else ""
                if tag in VI_KEEP_TAGS and len(word) > 1:
                    result.append(word)
            return result
        except Exception as e:
            print(f"  [underthesea error] {e} — fallback")

    # Fallback: lấy tất cả từ dài hơn 1 ký tự, bỏ stopwords cơ bản
    VI_STOPWORDS = {
        "là", "của", "và", "trong", "có", "được", "các", "một",
        "cho", "với", "từ", "đã", "về", "theo", "không", "này",
        "những", "để", "bị", "tại", "sẽ", "trên", "khi", "đó",
        "ra", "vào", "lên", "xuống", "thì", "mà", "rằng",
    }
    tokens = text.split()
    return [
        t for t in tokens
        if len(t) > 1
        and t.lower() not in VI_STOPWORDS
        and re.search(r"[a-zA-ZÀ-ỹ]", t)
    ]


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

    # Trích xuất content words
    src_words = extract_en_content_words(source)
    tgt_words = extract_vi_content_words(hypothesis)

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