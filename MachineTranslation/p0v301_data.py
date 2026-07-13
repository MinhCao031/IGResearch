from huggingface_hub import hf_hub_download
import csv

repo_id = "nhuvo/MedEV"

en_path = hf_hub_download(repo_id=repo_id, filename="test.en.new.txt", repo_type="dataset")
vi_path = hf_hub_download(repo_id=repo_id, filename="test.vi.new.txt", repo_type="dataset")

with open(en_path, encoding="utf-8") as f:
    en_lines = [l.strip() for l in f if l.strip()]

with open(vi_path, encoding="utf-8") as f:
    vi_lines = [l.strip() for l in f if l.strip()]

assert len(en_lines) == len(vi_lines), f"{len(en_lines)} vs {len(vi_lines)}"

with open("rp1v302_medev_envi_test.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "reference"])
    writer.writeheader()
    for en, vi in zip(en_lines, vi_lines):
        writer.writerow({"source": en, "reference": vi})

print(f"Saved {len(en_lines)} rows")