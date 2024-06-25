# IntelArcA770

Samples running deep learning models on Intel GPU Arc A770 on Windows, natively using Intel extension for PyTorch.

## Samples

- [llama2](./src/llama2) - Inference & Chat with Llama-2-7b-hf model
- [llama3](./src/llama3) - Inference & Chat with Llama-3-8B model
- [phi3](./src/phi3) - Chat with Phi-3-mini-4k-instruct model

## SetUp

Install [Intel extension for pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)

Install Intel graphics driver
Install oneAPI base toolkit
Install mamba or conda

```pwsh
mamba create -n arcA770 python=3.11
mamba activate arcA770
Install-Module Pscx -Scope CurrentUser -AllowClobber
Invoke-BatchFile 'C:\Program Files (x86)\Intel\oneAPI\setvars.bat'
mamba install pkg-config libuv
python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
python -m pip install setuptools==69.5.1
pip install numpy==1.26.4
```

Sanity Test

```pwsh
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

### Install transformers and other dependencies

To get hugging face token, login to hugging face and get the token from settings

```pwsh
python -m pip install transformers==4.37.0
pip install accelerate
pip install sentencepiece
huggingface-cli login
```

### Request access to meta-llama2

Go to <https://huggingface.co/meta-llama/Llama-2-7b-hf> and request access.

Llama3: https://huggingface.co/meta-llama/Meta-Llama-3-8B
CodeLlama: https://huggingface.co/meta-llama/CodeLlama-7b-hf

### Only if you are testing phi3

```pwsh
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.37.0
```

## Run

Modify the prompts as desired, then run the command

```pwsh
mamba activate arcA770
./setup_vars.ps1
python -W ignore ./src/phi3/chat.py
```
