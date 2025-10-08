from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id="amgadhasan/phi-2",repo_type="model", local_dir="./phi-2", local_dir_use_symlinks=False)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# We need to trust remote code since this hasn't been integrated in transformers as of version 4.35
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

def generate(prompt: str, generation_params: dict = {"max_length":200})-> str : 
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, **generation_params)
    completion = tokenizer.batch_decode(outputs)[0]
    return completion

result = generate("Write a poem about the moon.")
print(result)
