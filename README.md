# GPS: Gold Points Sniper for Fine-Grained Human Action Recognition
---

# Data Annotation
The script of data annotation using OV-72B
```
GPS-code/Data_Generation/inference_llava_next.py
```

Construct the instruction tuning dataset (multi-round conversation format) in LLaVA format. Such format is applied to fine-tune LLaVA-NeXT(OV-7B, NeXT-8B) and MiniGPT-v2
```
GPS-code/Data_Generation/starwar_llava.py
```

Construct the instruction tuning dataset (multi-round conversation format) in Qwen-VL format. Such format is applied to fine-tune Qwen-VL-Chat
```
GPS-code/Data_Generation/starwar_qwen.py
```

# LVLMs SGPS Fine-Tuning
Follow the original repo of LVLMs to deploy model components and training datasets, set up environments and launch the training pipeline.

[LLaVA-NeXT (OV-72B, NeXT-8B, OV-7B)](https://github.com/LLaVA-VL/LLaVA-NeXT)

[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)

[MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)


# LVLMs SGPS Structured Prompting
The corresponding scripts are organized in 
```
GPS-code/SGPS_infer
```
There is a directory containing scripts of various prompting structure (direct, SPGS, ablated) for each LVLM type.
# GPSE Multimodal Evaluation
Comprehensive Visual Understanding Evaluation by OV-72B
```
GPS-code/GPSE_Eval/OV_eval.py
```

Entailment Relationship Evaluation by Llama-3.3-70B-Instruct
Local Running (for free but need local deployment)
```
GPS-code/GPSE_Eval/llama_eval_sum_prop.py
```

Invoking API (cost money but much cheaper than GPT)
```
GPS-code/GPSE_Eval/api_llama_sum_prop.py
```
