import torch
from pathlib import Path
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

RAW = Path(
    "/mnt/aix22303/data/how2sign/How2Sign/prepared/"
    "mmslt_random4000_v3/descriptions/raw/"
    "how2sign_SLdescriptions.dev.raw"
)

OUT = Path(
    "/mnt/aix22303/data/how2sign/How2Sign/prepared/"
    "mmslt_random4000_v3/descriptions/"
    "how2sign_SLdescriptions.dev"
)

# ↓ 여기에 추가
def extract_text(v):
    if isinstance(v, str):
        return v

    if isinstance(v, dict):
        for key in ["texts", "text", "description", "descript", "caption"]:
            if key in v and isinstance(v[key], str):
                return v[key]

    raise TypeError(
        f"Unsupported description type: {type(v)}, value={v}"
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device:", device)
print("loading:", RAW)

data = torch.load(RAW, map_location="cpu")

print("samples:", len(data))

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")
model.to(device)
model.eval()

out = defaultdict(dict)

for k, v in tqdm(data.items()):
    texts = v["texts"]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )

        # CLS embedding
        feat = outputs.last_hidden_state[:, 0, :]

    out[k]["texts"] = texts
    out[k]["bert_feat"] = feat.cpu()

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(out, OUT)

print("saved:", OUT)
print("samples:", len(out))
