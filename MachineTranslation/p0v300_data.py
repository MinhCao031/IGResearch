from datasets import load_dataset
import csv

dataset  = load_dataset("IWSLT/mt_eng_vietnamese", "vi-en", trust_remote_code=True)
test_set = dataset["test"]

with open("rp1v301_iwslt15_envi.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "reference"])
    writer.writeheader()
    for row in test_set:
        t = row["translation"]
        writer.writerow({"source": t["en"], "reference": t["vi"]})

print(f"Saved {len(test_set)} rows")