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
start_time = time.time()
with torch.inference_mode(), torch.no_grad(), torch.autocast(
    ############# code changes ###############
    device_type="xpu",
    ##########################################
    enabled=True,
    dtype=dtype,
):
    text = "You may have heard of Evergine, a powerful component-based industrial engine developed on .NET and designed to be completely multiplatform."
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to("xpu")
    ##########################################
    generated_ids = model.generate(
        input_ids, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id
    )[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_text, "...")
print("Time taken:", time.time() - start_time, "seconds")
