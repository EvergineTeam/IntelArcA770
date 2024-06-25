import torch
import time

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Prompt for https://huggingface.co/microsoft/Phi-3-mini-4k-instruct#chat-format
PHI3_PROMPT_FORMAT = "<|user|>\n{prompt}<|end|>\n<|assistant|>"

model_id = "microsoft/Phi-3-mini-4k-instruct"
_prompt = "Would you recommend Evergine to create graphics-rich web applications based on WASM?"
_max_tokens = 64

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    trust_remote_code=True,
    optimize_model=True,
    use_cache=True,
    # # cpu_embedding=True
)

model = model.to("xpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Generate predicted tokens
with torch.inference_mode():
    prompt = PHI3_PROMPT_FORMAT.format(prompt=_prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("xpu")

    # ipex_llm model needs a warmup, then inference time can be accurate
    output = model.generate(input_ids, max_new_tokens=_max_tokens)
    # start inference
    st = time.time()

    output = model.generate(input_ids, do_sample=False, max_new_tokens=_max_tokens)
    torch.xpu.synchronize()
    end = time.time()
    output_str = tokenizer.decode(output[0], skip_special_tokens=False)
    print(f"Inference time: {end-st} s")
    print("-" * 20, "Prompt", "-" * 20)
    print(prompt)
    print("-" * 20, "Output", "-" * 20)
    print(output_str)
