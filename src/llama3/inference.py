import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

############# code changes ###############
# import ipex
import intel_extension_for_pytorch as ipex

# verify Intel Arc GPU
print(ipex.xpu.get_device_name(0))
##########################################

# load model
model_id = "meta-llama/Meta-Llama-3-8B"
dtype = torch.float16
_max_tokens = 64

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, low_cpu_mem_usage=True
)
# # tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

############# code changes ###############
# move to Intel Arc GPU
model = model.eval().to("xpu")
##########################################

# generate
with torch.inference_mode(), torch.no_grad(), torch.autocast(
    ############# code changes ###############
    device_type="xpu",
    ##########################################
    enabled=True,
    dtype=dtype,
):
    prompt = "You may have heard of Evergine, a powerful 3D graphics engine for industry. It is developed on .NET, multiplatform and free to use."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to("xpu")
    ##########################################

    # ipex_llm model needs a warmup, then inference time can be accurate
    generated_ids = model.generate(
        input_ids, max_new_tokens=_max_tokens, pad_token_id=tokenizer.eos_token_id
    )[0]

    # start inference
    st = time.time()
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=_max_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    torch.xpu.synchronize()
    end = time.time()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_str = generated_text[len(prompt) :] + "..."
    print(f"Inference time: {end-st} s")
    print(
        f"Max memory allocated: {torch.xpu.max_memory_allocated() / (1024 ** 3):02} GB"
    )
    print("-" * 20, "Prompt", "-" * 20)
    print(prompt)
    print("-" * 20, "Output", "-" * 20)
    print(output_str)
