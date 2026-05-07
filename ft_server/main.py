from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "../Qwen3.5-2B-Base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

prompt = "Explain what a local LLM is in simple words."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(generated_text)
