import requests, tarfile, io, csv

url = "https://github.com/stefan-it/nmt-en-vi/raw/master/data/test-2013-en-vi.tgz"
r = requests.get(url, timeout=30)
r.raise_for_status()

tar = tarfile.open(fileobj=io.BytesIO(r.content))
en_text = tar.extractfile("tst2013.en").read().decode("utf-8")
vi_text = tar.extractfile("tst2013.vi").read().decode("utf-8")

en_lines = [l for l in en_text.split("\n") if l.strip()]
vi_lines = [l for l in vi_text.split("\n") if l.strip()]
assert len(en_lines) == len(vi_lines)

with open("rp1v301_iwslt15_envi_test.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "reference"])
    writer.writeheader()
    for en, vi in zip(en_lines, vi_lines):
        writer.writerow({"source": en, "reference": vi})

print(f"Saved {len(en_lines)} rows")