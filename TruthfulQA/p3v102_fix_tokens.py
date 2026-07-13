"""
Post-processing tool for source_token and target_token fields.
Only replaces unmatched tokens — limit enforcement is handled upstream.

source_token: each word must appear verbatim in question
target_token: each word must appear verbatim in qwen_response

Replacement strategy (in order):
1. Group match        — same semantic group (e.g. negation, legality)
2. Morphological match — same root, same meaning (inflection / plural / suffix)
3. Char similarity    — character-level similarity fallback
4. Drop               — no suitable replacement found
"""

import json
import re
import sys
from difflib import SequenceMatcher

MIN_SCORE = 0.5

# ── semantic groups ───────────────────────────────────────────────────────────

GROUPS: list[set[str]] = [
    # negation
    {
        "not", "no", "never", "nor", "cannot", "neither",
        "none", "nothing", "nowhere", "nobody",
        "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
        "won't", "wont", "can't", "cant", "isn't", "isnt",
        "aren't", "arent", "wasn't", "wasnt", "weren't", "werent",
        "hasn't", "hasnt", "haven't", "havent", "hadn't", "hadnt",
        "wouldn't", "wouldnt", "couldn't", "couldnt", "shouldn't", "shouldnt",
    },
    # affirmation
    {
        "yes", "yeah", "yep", "correct", "right", "true",
        "indeed", "certainly", "absolutely", "definitely",
    },
    # fictional / unreal
    {
        "fictional", "fictitious", "fake", "false", "fabricated",
        "mythical", "mythological", "legendary", "imaginary",
        "unreal", "nonexistent", "invented", "made-up",
    },
    # unproven / discredited
    {
        "unproven", "unverified", "unsupported", "unfounded",
        "disputed", "debunked", "discredited", "disproven",
        "inconclusive", "unreliable", "pseudoscientific",
    },
    # proven / confirmed
    {
        "proven", "confirmed", "verified", "validated",
        "established", "demonstrated", "supported", "accepted",
    },
    # legal / permitted
    {
        "legal", "lawful", "permitted", "allowed",
        "authorized", "permissible", "valid",
    },
    # illegal / prohibited
    {
        "illegal", "unlawful", "banned", "prohibited",
        "forbidden", "restricted", "outlawed", "criminalized",
    },
    # variation / context-dependence
    {
        "varies", "vary", "variable", "depends", "dependent",
        "differ", "differs", "inconsistent", "diverse", "mixed",
    },
    # universality — all
    {
        "all", "every", "everyone", "everything", "everywhere",
        "always", "universal", "universally", "entirely",
        "completely", "wholly", "throughout",
    },
    # harm / danger
    {
        "harmful", "dangerous", "hazardous", "toxic",
        "damaging", "injurious", "unsafe", "deadly", "lethal",
    },
    # safe / harmless
    {
        "safe", "harmless", "benign", "risk-free",
        "non-toxic", "innocuous",
    },
]

def find_group(word: str) -> set[str] | None:
    w = word.lower()
    for group in GROUPS:
        if w in group:
            return group
    return None


# ── morphological helpers ─────────────────────────────────────────────────────

# Prefixes that reverse meaning — never match across these
_NEGATING_PREFIXES = frozenset({
    "un", "in", "il", "im", "ir", "dis", "non", "mis", "anti",
})

# (suffix_to_strip, min_chars_remaining, suffix_to_add_back)
# Ordered longest-first; -ful/-less excluded (polarity-changing)
_SUFFIX_RULES: list[tuple[str, int, str]] = [
    ("ization", 3, ""),
    ("isation", 3, ""),
    ("nesses",  3, ""),
    ("ments",   3, ""),
    ("tions",   3, ""),
    ("ness",    3, ""),
    ("ment",    3, ""),
    ("tion",    3, ""),
    ("sion",    3, ""),
    ("ies",     2, "y"),   # cities  → city
    ("ied",     2, "y"),   # tried   → try
    ("ing",     3, ""),    # proving → prov
    ("ing",     3, "e"),   # proving → prove
    ("ize",     3, ""),    # legalize → legal
    ("ise",     3, ""),
    ("ity",     3, ""),    # legality → legal
    ("ive",     3, ""),
    ("ous",     3, ""),
    ("ial",     3, ""),
    ("al",      4, ""),    # musical → music
    ("ly",      4, ""),    # quickly → quick
    ("ed",      3, ""),    # proved  → prov
    ("ed",      3, "e"),   # proved  → prove
    ("er",      3, ""),    # faster  → fast
    ("es",      3, ""),
    ("es",      3, "e"),
    ("s",       3, ""),    # runs    → run
]


def get_roots(word: str) -> set[str]:
    """Return all plausible stems of *word* via suffix stripping."""
    w = word.lower()
    results = {w}
    for suffix, min_len, add_back in _SUFFIX_RULES:
        if w.endswith(suffix) and len(w) - len(suffix) >= min_len:
            base = w[:-len(suffix)]
            # doubled consonant before -ing/-ed: running → run
            if not add_back and suffix in ("ing", "ed"):
                if len(base) >= 2 and base[-1] == base[-2]:
                    results.add(base[:-1])
            results.add(base + add_back)
    return results


def is_negated_variant(a: str, b: str) -> bool:
    """True if one word is the prefix-negated form of the other."""
    a, b = a.lower(), b.lower()
    for prefix in _NEGATING_PREFIXES:
        if (a.startswith(prefix) and a[len(prefix):] == b) or \
           (b.startswith(prefix) and b[len(prefix):] == a):
            return True
    return False


def morphological_match(token: str, candidates: list[str]) -> str | None:
    """Find a candidate that is a same-meaning morphological variant of *token*."""
    token_roots = get_roots(token)
    for cand in candidates:
        if is_negated_variant(token, cand):
            continue
        if token_roots & get_roots(cand):
            return cand
    return None


# ── string helpers ────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text)


def is_verbatim(word: str, text: str) -> bool:
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best_similarity_match(word: str, candidates: list[str]) -> str | None:
    best_word, best_score = None, 0.0
    for cand in candidates:
        score = char_similarity(word, cand)
        if score > best_score:
            best_score, best_word = score, cand
    return best_word if best_score >= MIN_SCORE else None


# ── core logic ────────────────────────────────────────────────────────────────

def find_replacement(token: str, candidates: list[str]) -> str | None:
    # 1. group match
    group = find_group(token)
    if group:
        for cand in candidates:
            if cand.lower() in group:
                return cand

    # 2. morphological match
    match = morphological_match(token, candidates)
    if match:
        return match

    # 3. char similarity
    return best_similarity_match(token, candidates)


def fix_tokens(tokens: list[str], anchor: str) -> list[str]:
    # Dedupe by lowercase but preserve original casing of first occurrence
    seen: set[str] = set()
    candidates: list[str] = []
    for w in tokenize(anchor):
        if w.lower() not in seen:
            seen.add(w.lower())
            candidates.append(w)
    result = []
    for token in tokens:
        if is_verbatim(token, anchor):
            result.append(token)
        else:
            replacement = find_replacement(token, candidates)
            if replacement is not None:
                result.append(replacement)
            # else: drop
    return result


def process_item(item: dict) -> dict:
    item["source_tokens"] = fix_tokens(item["source_tokens"], item["question"])
    item["target_token"]  = fix_tokens(item["target_token"],  item["qwen_response"])
    return item


def process(data: list[dict]) -> list[dict]:
    for i, item in enumerate(data):
        if "tokens_status" not in item:
            raise ValueError(f"Item {i}: missing 'tokens_status'")
        if item["tokens_status"] != 0:
            data[i] = process_item(item)
        # Remove status fields from all items
        data[i].pop("qtoken_status", None)
        data[i].pop("atoken_status", None)
    return data

# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "wp2v104_qa_tokens.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "wp3v104_fix_tokens.json"

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    fixed = process(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(fixed)} items → {output_path}")
