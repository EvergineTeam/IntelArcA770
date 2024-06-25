import datetime
import time
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

############# code changes ###############
# import ipex
import intel_extension_for_pytorch as ipex

# verify Intel Arc GPU
print(ipex.xpu.get_device_name(0))
##########################################

# load model
model_id = "meta-llama/Llama-2-7b-chat-hf"
dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, low_cpu_mem_usage=True
)
tokenizer = LlamaTokenizer.from_pretrained(model_id)

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
    text = f"<s>[INST] <<SYS>>You are Evergine AI, a friendly AI Assistant. Today's date is {datetime.date.today().strftime('%A, %B %d, %Y')}. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. The user lives in Spain, so be aware of the local context and preferences. Use a conversational tone and provide helpful and informative responses, utilizing external knowledge when necessary<</SYS>> Would you recommend Evergine to create graphics-rich web applications based on WASM? [/INST]"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to("xpu")
    ##########################################
    generated_ids = model.generate(input_ids, max_new_tokens=512)[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_text)
print("Time taken:", time.time() - start_time, "seconds")
