# SLM-based Local Agent
This is a personal agent for low-spec computer users. I set the prompt to respond in Korean.

### My laptop spec
```
- OS: Windows 11 Pro x64
- CPU: AMD Ryzen 7 PRO 6850U
- RAM: LPDDR5 16.0GB
- GPU: AMD Radeon(TM) Graphics 2.0GB
- Storage: 512GB SSD
```
It works on CPU only and automatically detects the number of available CPU cores.

## Requirements

### Dependency
This project uses the **llama-cpp-python** library, which executes local models via llama.cpp. Therefore, a C++ compiler must be installed on your system.

### Model
Before running, you need to download a GGUF model file into the './model' directory.
I downloaded **Qwen2.5-7B-Instruct-Q5_K_M.gguf** model from [Huggingface](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF).

Also, this code is optimized for Qwen-formatted prompts and tool calling methods. Other models can be run, but they may not function as well.