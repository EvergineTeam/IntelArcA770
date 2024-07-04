import datetime
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dtype = torch.float16
_max_tokens = 256

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

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
    messages = [
        {
            "role": "system",
            "content": f"You are Evergine AI, a friendly AI Assistant. Today's date is {datetime.date.today().strftime('%A, %B %d, %Y')}. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. The user lives in Spain, so be aware of the local context and preferences. Use a conversational tone and provide helpful and informative responses, utilizing external knowledge when necessary",
        },
        {
            "role": "user",
            "content": "Would you recommend Evergine to create graphics-rich web applications based on WASM?",
        },
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to(device_type)
    ##########################################

    # ipex_llm model needs a warmup, then inference time can be accurate
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=_max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        temperature=0.6,
        top_p=0.9,
    )[0]

    # start inference
    st = time.time()
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=_max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        temperature=0.6,
        top_p=0.9,
    )[0]
    if xpu_available:
        torch.xpu.synchronize()
    elif cuda_available:
        torch.cuda.synchronize()
    else:
        torch.cpu.synchronize()
    end = time.time()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    prompt_length = sum(len(m["role"]) + len(m["content"]) + 2 for m in messages)
    output_str = generated_text[prompt_length:] + "..."
    print(f"Inference time: {end-st} s")
    max_memory = (
        torch.cuda.max_memory_allocated()
        if cuda_available
        else torch.xpu.max_memory_allocated()
    )
    print(f"Max memory allocated: {max_memory / (1024 ** 3):02} GB")
    print("-" * 20, "Prompt", "-" * 20)
    print(f"System: {messages[0]['content']}\nUser: {messages[1]['content']}")
    print("-" * 20, "Output", "-" * 20)
    print(output_str)
