import time
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate.utils import is_xpu_available, is_ipex_available

cuda_available = torch.cuda.is_available()
xpu_available = is_xpu_available() and is_ipex_available()

############# code changes ###############
if xpu_available:
    import intel_extension_for_pytorch as ipex

    # verify Intel Arc GPU
    print(ipex.xpu.get_device_name(0))
##########################################

# load model
model_id = "meta-llama/Llama-2-7b-hf"
dtype = torch.float16
_max_tokens = 128

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, low_cpu_mem_usage=True
)
tokenizer = LlamaTokenizer.from_pretrained(model_id)

############# code changes ###############
device_type = "cpu"
if xpu_available:
    device_type = "xpu"
elif cuda_available:
    device_type = "cuda"

model = model.eval().to(device_type)
##########################################

# generate
with torch.inference_mode(), torch.no_grad(), torch.autocast(
    ############# code changes ###############
    device_type=device_type,
    ##########################################
    enabled=True,
    dtype=dtype,
):
    prompt = "You may have heard of Evergine, a powerful component-based industrial engine developed on .NET and designed to be completely multiplatform. With it, you can deploy to any of the platforms we support, and many more are to come! From Windows/PC applications to Web platforms (using WASM), passing through mobile devices, your Evergine applications has the potential to reach all of your users."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to(device_type)
    ##########################################

    # ipex_llm model needs a warmup, then inference time can be accurate
    output = model.generate(input_ids, max_new_tokens=_max_tokens)

    # start inference
    st = time.time()
    generated_ids = model.generate(
        input_ids, do_sample=False, max_new_tokens=_max_tokens
    )[0]
    if xpu_available:
        torch.xpu.synchronize()
    elif cuda_available:
        torch.cuda.synchronize()
    else:
        torch.cpu.synchronize()
    end = time.time()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_str = generated_text[len(prompt) :] + "..."
    print(f"Inference time: {end-st} s")
    max_memory = (
        torch.cuda.max_memory_allocated()
        if cuda_available
        else torch.xpu.max_memory_allocated()
    )
    print(f"Max memory allocated: {max_memory / (1024 ** 3):02} GB")
    print("-" * 20, "Prompt", "-" * 20)
    print(prompt)
    print("-" * 20, "Output", "-" * 20)
    print(output_str)
