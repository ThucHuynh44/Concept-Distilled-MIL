import os
import torch
from transformers import AutoModel, AutoTokenizer
from prompts import brca_prompts,nsclc_prompts,rcc_prompts  # hoặc nsclc_prompts, rcc_prompts

save_path = "./prompt_feats_titan/"
os.makedirs(save_path, exist_ok=True)

# 1️⃣ Load TITAN model & tokenizer
titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)

# 2️⃣ Load prompt templates
cls_templates = rcc_prompts()
text_feats = []

# 3️⃣ Encode text with TITAN
for i in range(len(cls_templates)):
    inputs = tokenizer(cls_templates[i], return_tensors="pt", padding=True).to("cuda")

    # 👉 TITAN chỉ nhận input_ids
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        text_emb = titan.encode_text(inputs["input_ids"])

    text_feats.append(text_emb.detach().cpu().float())

# 4️⃣ Save features
torch.save(text_feats, os.path.join(save_path, "rcc_concepts_titan.pt"))
print("✅ Saved TITAN text features successfully!")
