import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import IntegratedGradients
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# =========================
# 1. CẤU HÌNH
# =========================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """You are an intelligent, conversational assistant dedicated to providing the absolute truth.

Your task:
- Answer in one sentence where possible; use two sentences only if a single sentence would be misleading or incomplete.
- Keep the sentence concise: use only the words necessary to make the point.
- If the question is a yes/no question, unless the opinion is uncertain, always begin your answer with "Yes" or "No".
- If the question describes an observable phenomenon (something that genuinely appears or behaves a certain way from a specific perspective), prioritize explaining it within that perspective.
- If the question relies on something entirely untrue with no observable basis, gently correct the assumption right away.
- If the information is uncertain or unavailable, say so explicitly rather than guessing."""

# Kiểm soát phần nào của prompt được report attribution
ATTR_SYSTEM   = False
ATTR_QUESTION = True
ATTR_ANSWER   = False

INPUT_JSON  = "wp3v104_fix_tokens.json"
OUTPUT_JSON = "wp4v101_qa_ig.json"

count = 0
require_count = 900

# =========================
# 2. LOAD MODEL
# =========================
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Loading {MODEL_ID} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
    device_map="cuda",
)
model.eval()
model.config.use_cache = False  # bắt buộc cho Captum

# Tính offset cuối system prompt (= đầu user/question block)
# Dùng empty user turn để đo đúng độ dài phần system
_prefix_text = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user",   "content": ""}],
    tokenize=False,
    add_generation_prompt=False,
)
QUESTION_START = len(tokenizer(_prefix_text, return_tensors="pt").input_ids[0])

print(f"System prompt token length : {QUESTION_START}")
print(f"Attribution config         : "
      f"system={ATTR_SYSTEM}, question={ATTR_QUESTION}, answer={ATTR_ANSWER}")


# =========================
# 3. HÀM TÌM VỊ TRÍ TOKEN
# =========================
def find_token_position(target_str: str, answer_ids, prompt_len: int):
    """Tìm vị trí tuyệt đối của target_str trong answer_ids."""
    target_lower = target_str.lower().strip()
    for i, tid in enumerate(answer_ids):
        tok_str = tokenizer.decode([tid]).lower().strip()
        if target_lower in tok_str or tok_str in target_lower:
            return prompt_len + i, tid.item()
    return -1, -1


# =========================
# 4. HÀM TÍNH IG
# =========================
def compute_ig_for_target(
    target_pos: int,
    target_token_id: int,
    context_ids,
    prompt_ids,
    prompt_len: int,
) -> list[dict]:
    """Tính Integrated Gradients cho một target token.
    Trả về list {"t": str, "s": float} cho các segment được bật."""

    ig_context_ids  = context_ids[:, :target_pos + 1]
    ig_context_mask = torch.ones_like(ig_context_ids, device=device)

    def forward_func(embeds):
        with torch.autocast(device_type="cuda", dtype=model_dtype):
            logits = model(
                inputs_embeds=embeds,
                attention_mask=ig_context_mask,
            ).logits
        return logits[:, -1, :].float()

    ig = IntegratedGradients(forward_func)
    context_embeds = (
        model.get_input_embeddings()(ig_context_ids)
        .detach()
        .requires_grad_(True)
    )
    baseline = torch.zeros_like(context_embeds)

    attributions = ig.attribute(
        context_embeds,
        baselines=baseline,
        target=target_token_id,
        n_steps=24,
        internal_batch_size=1,
    )

    raw_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Xác định slice cần report theo config
    segments = []
    if ATTR_SYSTEM:
        segments.append((0, QUESTION_START - 2))
    if ATTR_QUESTION:
        segments.append((QUESTION_START - 2, prompt_len))
    if ATTR_ANSWER:
        segments.append((prompt_len, target_pos + 1))

    attribution_data = []
    for seg_start, seg_end in segments:
        for tok_id, score in zip(
            prompt_ids[seg_start:seg_end],
            raw_scores[seg_start:seg_end],
        ):
            clean_str = (
                tokenizer.decode([tok_id])
                .replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .strip()
            )
            if clean_str:
                attribution_data.append({"t": clean_str, "s": float(score)})

    del context_embeds, attributions, baseline, ig_context_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return attribution_data


# =========================
# 5. HÀM XỬ LÝ MỘT MẪU
# =========================
def explain_instance(item: dict):

    question      = item["question"]
    qwen_response = item["qwen_response"]
    source_tokens = item["source_tokens"]
    raw_target  = item["target_token"]
    target_list = raw_target if isinstance(raw_target, list) else [raw_target]

    if item.get("tokens_status") != 0:
        print(f"Fixed {item.get('id', '?')}", question, qwen_response, source_tokens, raw_target,
              "--------------------------------------------------------------------",sep="\n")

    # Tái tạo prompt gốc
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize một lần, dùng chung cho tất cả target_token
    prompt_ids  = tokenizer(prompt_text, return_tensors="pt").input_ids[0].to(device)
    prompt_len  = prompt_ids.shape[0]
    answer_ids  = tokenizer(
        qwen_response, return_tensors="pt", add_special_tokens=False
    ).input_ids[0].to(device)
    context_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0)

    ig_list = []

    for target_str in target_list:

        # Kiểm tra substring
        if target_str.lower() not in qwen_response.lower():
            print(
                f"  [Warning] '{target_str}' không phải substring "
                f"của response ID {item['id']}: {qwen_response}"
            )

        # Tìm vị trí token
        target_pos, target_token_id = find_token_position(
            target_str, answer_ids, prompt_len
        )
        if target_pos == -1:
            print(
                f"  [Skipped] Không tìm thấy token <{target_str}> "
                f"trong answer_ids ID {item['id']}: <{qwen_response}>"
            )
            continue

        # Tính IG
        attribution_data = compute_ig_for_target(
            target_pos, target_token_id,
            context_ids, prompt_ids, prompt_len,
        )

        ig_list.append({
            "token":               target_str,
            "target_pos_absolute": target_pos,
            "attribution_scores":  attribution_data,
        })

    if not ig_list:
        return None

    return {
        "id":            item["id"],
        "question":      question,
        "qwen_response": qwen_response,
        "source_tokens": source_tokens,
        "target_tokens": target_list,
        "ig":            ig_list,
    }


# =========================
# 6. VÒNG LẶP XỬ LÝ
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"\nBắt đầu chạy Integrated Gradients cho {len(dataset)} mẫu...")
results = []
skipped = 0

for item in tqdm(dataset):
    try:
        count += 1
        if count > require_count:
            break
        result = explain_instance(item)
        if result:
            results.append(result)
        else:
            print(f"  Bỏ qua ID {item.get('id', '?')}: không có target token hợp lệ")
            skipped += 1
    except Exception as e:
        print(f"  Lỗi crash ở ID {item.get('id', '?')}: {e}")
        skipped += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nHoàn thành!")
print(f"  Mẫu đầu vào:   {len(dataset)}")
print(f"  Object đầu ra: {len(results)}  (1 object / id)")
print(f"  Bị bỏ qua:     {skipped}")
print(f"  File kết quả:  {OUTPUT_JSON}")