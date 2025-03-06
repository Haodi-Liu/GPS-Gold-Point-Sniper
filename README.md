# GPS: Gold Points Sniper for Fine-Grained Human Action Recognition
---

# Data Annotation
The script of data annotation
```
GPS-code/Data_Generation/inference_llava_next.py
```
# LVLMs SGPS Fine-Tuning
Follow the original repo of LVLMs to deploy model components and datasets, set up environments and launch the training pipeline.

[LLaVA-NeXT (OV72B, NeXT8B, OV7B)](https://github.com/LLaVA-VL/LLaVA-NeXT)

[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)

[MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)


# LVLMs SGPS Structured Prompting
The corresponding scripts are organized in 
```
GPS-code/SGPS_infer
```

# GPSE Multimodal Evaluation
Comprehensive Visual Understanding Evaluation by OV72B
```
GPS-code/GPSE_Eval/OV_eval.py
```

Entailment Relationship Evaluation by Llama-3.3-70B
Local Running (for free but need local deployment)
```
GPS-code/GPSE_Eval/llama_eval_sum_prop.py
```

Invoking API (cost money but much cheaper than GPT)
```
GPS-code/GPSE_Eval/api_llama_sum_prop.py
```
