import os
import json
import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from captum.attr import IntegratedGradients
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# =========================
# Config
# =========================
MODEL_ID   = "Helsinki-NLP/opus-mt-en-vi"
INPUT_JSON  = "wp2v301_mt_tokens.json"   # output của P2 (sau khi có)
OUTPUT_JSON = "wp4v301_mt_ig.json"
N_STEPS     = 24

# =========================
# Device
# =========================
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# Load model
# =========================
print(f"Loading {MODEL_ID}...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
model     = MarianMTModel.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
).to(device)
model.eval()
# use_cache = False bắt buộc cho Captum
model.config.use_cache = False
print(f"Model loaded. Parameters: {model.num_parameters()/1e6:.1f}M")


# =========================
# Forward function — sequence-level log-prob
# =========================
def make_forward_func(decoder_input_ids, attention_mask, decoder_attention_mask):
    """
    Trả về forward_func nhận encoder_embeds và tính
    tổng log-probability của toàn bộ hypothesis sequence.

    F(x) = sum_t log P(t | x, t_<t)

    Đây là target scalar để IG tính gradient theo encoder embedding.
    """
    def forward_func(encoder_embeds):
        with torch.autocast(device_type="cuda", dtype=model_dtype):
            outputs = model(
                inputs_embeds=encoder_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
        # logits: [1, tgt_len, vocab_size]
        logits = outputs.logits.float()

        # Log-softmax → log-prob của từng vị trí
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Lấy log-prob của token thực tế tại mỗi vị trí
        # decoder_input_ids là [BOS, t1, t2, ...], labels là [t1, t2, ..., EOS]
        # Shift: logits[i] dự đoán token [i+1]
        labels    = decoder_input_ids[:, 1:]        # bỏ BOS
        log_probs = log_probs[:, :-1, :]            # bỏ vị trí cuối

        # Gather log-prob của token đúng tại mỗi vị trí
        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)   # [1, tgt_len - 1]

        # Tổng log-prob = F(x) — scalar
        return token_log_probs.sum(dim=-1)   # [1]

    return forward_func


# =========================
# Tính IG cho 1 mẫu
# =========================
def compute_ig(item: dict) -> dict | None:
    source     = item["source"]
    hypothesis = item["hypothesis"]

    if not source or not hypothesis:
        return None

    # Tokenize source → encoder input
    enc = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(device)

    encoder_input_ids  = enc["input_ids"]
    attention_mask     = enc["attention_mask"]

    # Tokenize hypothesis → decoder input (thêm BOS ở đầu)
    tgt_ids = tokenizer(
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        add_special_tokens=False,
    ).input_ids.to(device)

    # MarianMT dùng pad_token_id làm BOS cho decoder
    bos_id  = torch.tensor([[tokenizer.pad_token_id]], device=device)
    decoder_input_ids  = torch.cat([bos_id, tgt_ids], dim=1)
    decoder_attention_mask = torch.ones_like(decoder_input_ids)

    # Encoder embeddings
    encoder_embeds = (
        model.get_encoder()
        .embed_tokens(encoder_input_ids)
        .detach()
        .requires_grad_(True)
    )

    # Nhân với positional scaling nếu model dùng (MarianMT dùng embed_scale)
    embed_scale = getattr(model.get_encoder(), "embed_scale", 1.0)
    encoder_embeds_scaled = encoder_embeds * embed_scale

    baseline = torch.zeros_like(encoder_embeds_scaled)

    forward_func = make_forward_func(
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
    )

    ig = IntegratedGradients(forward_func)

    try:
        attributions = ig.attribute(
            encoder_embeds_scaled,
            baselines=baseline,
            n_steps=N_STEPS,
            internal_batch_size=1,
        )
    except Exception as e:
        print(f"\n  [IG Error] ID {item['id']}: {e}")
        return None

    # Sum over embedding dim → scalar per source token
    raw_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    # raw_scores: [src_seq_len]

    # Map về token string
    src_token_ids  = encoder_input_ids[0].tolist()
    src_token_strs = [
        tokenizer.decode([tid]).strip()
        for tid in src_token_ids
    ]

    attribution_data = [
        {"t": tok, "s": float(score)}
        for tok, score in zip(src_token_strs, raw_scores)
        if tok and tok not in ("", "<pad>", "</s>")
    ]

    del encoder_embeds, encoder_embeds_scaled, attributions, baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "id":               item["id"],
        "source":           source,
        "hypothesis":       item["hypothesis"],
        "reference":        item.get("reference", ""),
        "source_tokens":    item.get("source_tokens", []),
        "target_tokens":    item.get("target_tokens", []),
        "attribution_scores": attribution_data,   # list[{t, s}] — 1 entry / source token
    }


# =========================
# Main loop
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"\nBắt đầu tính IG cho {len(dataset)} mẫu...")
print(f"N_STEPS = {N_STEPS}  (24 forward passes / câu)")

results = []
skipped = 0

for item in tqdm(dataset):
    try:
        result = compute_ig(item)
        if result:
            results.append(result)
        else:
            skipped += 1
    except Exception as e:
        print(f"\n  [Crash] ID {item.get('id', '?')}: {e}")
        skipped += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nHOÀN THÀNH → {OUTPUT_JSON}")
print(f"Tổng:       {len(dataset)}")
print(f"Thành công: {len(results)}")
print(f"Bỏ qua:     {skipped}")

# Kiểm tra nhanh output
if results:
    sample = results[0]
    print(f"\nVí dụ:")
    print(f"  Source: {sample['source'][:70]}")
    print(f"  Hypothesis: {sample['hypothesis'][:70]}")
    print(f"  Attribution scores ({len(sample['attribution_scores'])} tokens):")
    for a in sample["attribution_scores"][:5]:
        print(f"    '{a['t']}': {a['s']:.4f}")