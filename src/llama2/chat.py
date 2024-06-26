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
_max_tokens = 512

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, low_cpu_mem_usage=True
)
tokenizer = LlamaTokenizer.from_pretrained(model_id)

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
    prompt = f"<s>[INST] <<SYS>>You are Evergine AI, a friendly AI Assistant. Today's date is {datetime.date.today().strftime('%A, %B %d, %Y')}. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. The user lives in Spain, so be aware of the local context and preferences. Use a conversational tone and provide helpful and informative responses, utilizing external knowledge when necessary<</SYS>> Would you recommend Evergine to create graphics-rich web applications based on WASM? [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to("xpu")
    ##########################################

    # ipex_llm model needs a warmup, then inference time can be accurate
    generated_ids = model.generate(input_ids, max_new_tokens=_max_tokens)[0]

    # start inference
    st = time.time()
    generated_ids = model.generate(
        input_ids, do_sample=False, max_new_tokens=_max_tokens
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
