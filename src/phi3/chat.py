import torch
import time

from transformers import AutoTokenizer
from accelerate.utils import is_xpu_available, is_ipex_available

cuda_available = torch.cuda.is_available()
xpu_available = is_xpu_available() and is_ipex_available()

############# code changes ###############
if xpu_available:
    from ipex_llm.transformers import AutoModelForCausalLM
    import intel_extension_for_pytorch as ipex

    # verify Intel Arc GPU
    print(ipex.xpu.get_device_name(0))
else:
    from transformers import AutoModelForCausalLM
##########################################

# Prompt for https://huggingface.co/microsoft/Phi-3-mini-4k-instruct#chat-format
PHI3_PROMPT_FORMAT = "<|user|>\n{prompt}<|end|>\n<|assistant|>"

model_id = "microsoft/Phi-3-mini-4k-instruct"
_prompt = "Would you recommend Evergine to create graphics-rich web applications based on WASM?"
_max_tokens = 512

if xpu_available:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        trust_remote_code=True,
        optimize_model=True,
        use_cache=True,
        # # cpu_embedding=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        trust_remote_code=True,
        use_cache=True,
        # # cpu_embedding=True
    )

############# code changes ###############
device_type = "cpu"
if xpu_available:
    device_type = "xpu"
elif cuda_available:
    device_type = "cuda"

if xpu_available:
    model = model.to(device_type)
##########################################

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Generate predicted tokens
with torch.inference_mode():
    prompt = PHI3_PROMPT_FORMAT.format(prompt=_prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device_type)

    # ipex_llm model needs a warmup, then inference time can be accurate
    output = model.generate(input_ids, max_new_tokens=_max_tokens)
    # start inference
    st = time.time()

    output = model.generate(input_ids, do_sample=False, max_new_tokens=_max_tokens)
    if xpu_available:
        torch.xpu.synchronize()
    elif cuda_available:
        torch.cuda.synchronize()
    else:
        torch.cpu.synchronize()
    end = time.time()
    output_str = tokenizer.decode(output[0], skip_special_tokens=False)
    output_str = (
        output_str.replace("<|user|>", "")
        .replace("<|assistant|>", "")
        .replace("<|end|>", "")[len(_prompt) + 4 :]
    )
    print(f"Inference time: {end-st} s")
    max_memory = (
        torch.cuda.max_memory_allocated()
        if cuda_available
        else torch.xpu.max_memory_allocated()
    )
    print(f"Max memory allocated: {max_memory / (1024 ** 3):02} GB")
    print("-" * 20, "Prompt", "-" * 20)
    print(
        prompt.replace("<|user|>", "")
        .replace("<|assistant|>", "")
        .replace("<|end|>", "")
    )
    print("-" * 20, "Output", "-" * 20)
    print(output_str)
