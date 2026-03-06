# `finetune.sh` 真实依赖清单（静态追踪 + 路径核验）

## 1. 最重要结论
- 实际入口链是：`finetune.sh` -> `train.py` -> `minigpt4` 训练主链（task/dataset/model/runner）。
- 这次训练配置会命中 `capvid_frame` 数据集构建器与 `minigpt_v2` 模型，核心数据文件是 `cap_videos_14k_frame.json` 和其引用的图片。
- `cap_videos_14k_frame.json` 共 `14280` 条，全部是 `image` 样本（`video` 为 `0`），对应图片都存在（缺失 `0`）。
- 模型侧会读取：Llama2 主模型目录（HF 格式分片）、`checkpoint_stage3.pth`（项目内）以及 `eva_vit_g.pth`（通过 URL 缓存到本地后加载）。
- 你说明了 `/raid/hdliu` 等价 `/raid/hdliu_raid`。代码里数据默认写的是 `/raid/hdliu/...`，本文按你的要求统一映射到 `/raid/hdliu_raid/...` 做核验。

## 2. `~/MiniGPT-4` 下真正涉及的文件和脚本

## 2.1 直接被命令行引用（硬依赖）
- `finetune.sh`
- `train.py`
- `train_configs/minigptv2_finetune.yaml`

对应证据：
- `finetune.sh:3`。

## 2.2 配置解析与本次必经主链（实际会执行）
- 配置解析与合并：
- `minigpt4/common/config.py`
- `minigpt4/configs/models/minigpt_v2.yaml`
- `minigpt4/configs/datasets/llava/capvid_frame.yaml`

- 任务与数据主链：
- `minigpt4/tasks/__init__.py`
- `minigpt4/tasks/base_task.py`
- `minigpt4/tasks/image_text_pretrain.py`
- `minigpt4/datasets/builders/image_text_pair_builder.py`
- `minigpt4/datasets/datasets/llava_dataset.py`
- `minigpt4/processors/blip_processors.py`

- 模型主链：
- `minigpt4/models/minigpt_v2.py`
- `minigpt4/models/minigpt_base.py`
- `minigpt4/models/base_model.py`
- `minigpt4/models/modeling_llama.py`
- `minigpt4/models/eva_vit.py`

- 训练与保存：
- `minigpt4/common/optims.py`
- `minigpt4/runners/runner_base.py`

关键分支证据：
- `train.py:88-100`（build_datasets/build_model/runner.train）。
- `minigpt4/common/config.py:72-80,98-113`（从默认 model/dataset yaml 合并）。
- `minigpt4/datasets/builders/image_text_pair_builder.py:126-147`（`capvid_frame` 命中）。
- `minigpt4/datasets/datasets/llava_dataset.py:107-120`（读 `ann_path`，再按 `vis_root + image` 开图）。
- `minigpt4/models/minigpt_v2.py:133-137`（加载 `ckpt`）。
- `minigpt4/models/base_model.py:171-188`（`AutoTokenizer/LlamaForCausalLM.from_pretrained`）。
- `minigpt4/models/eva_vit.py:429-433`（下载/读取 `eva_vit_g.pth`）。
- `minigpt4/runners/runner_base.py:349-354,575-600,651-659`（输出目录、checkpoint、log）。

## 2.3 启动阶段 import 触发的附加依赖（会加载模块，但不一定跑到其业务分支）
- 包初始化：
- `minigpt4/__init__.py`
- `minigpt4/datasets/builders/__init__.py`
- `minigpt4/models/__init__.py`
- `minigpt4/processors/__init__.py`
- `minigpt4/runners/__init__.py`

- 因 `image_text_pair_builder.py` 顶层 import 一并加载的数据集模块：
- `minigpt4/datasets/datasets/laion_dataset.py`
- `minigpt4/datasets/datasets/cc_sbu_dataset.py`
- `minigpt4/datasets/datasets/text_caps.py`
- `minigpt4/datasets/datasets/llava_dataset.py`
- `minigpt4/datasets/datasets/unnatural_instruction.py`
- `minigpt4/datasets/datasets/multitask_conversation.py`
- `minigpt4/datasets/datasets/flickr.py`
- `minigpt4/datasets/datasets/vg_dataset.py`
- `minigpt4/datasets/datasets/coco_dataset.py`
- `minigpt4/datasets/datasets/gqa_datasets.py`
- `minigpt4/datasets/datasets/aok_vqa_datasets.py`
- `minigpt4/datasets/datasets/coco_vqa_datasets.py`
- `minigpt4/datasets/datasets/ocrvqa_dataset.py`
- `minigpt4/datasets/datasets/coco_caption.py`

- 因 `models/__init__.py` 顶层 import 一并加载：
- `minigpt4/models/minigpt4.py`
- `minigpt4/models/Qformer.py`

对应证据：
- `train.py:29-33`，`minigpt4/__init__.py:15-18`，`minigpt4/datasets/builders/image_text_pair_builder.py:7-20`，`minigpt4/models/__init__.py:14-16`。

## 2.4 当前配置下不会实际参与训练主分支的内容
- `eval_scripts/`、`eval_configs/`、`demo*.py`、`examples*/`、`prompts/`、绝大多数 `train_configs/*.yaml` 不在这条调用链。
- `MiniGPT4`（非 v2）相关分支与 Q-Former 训练路径不会在本次 `arch: minigpt_v2` 主分支中执行。

## 3. `/raid/hdliu_raid` 下真实涉及的数据集与模型组件

## 3.1 模型输入（读取）

### A. LLM 主模型目录
路径：`/raid/hdliu_raid/models/MiniGPT/Llama-2-7b-chat-hf`

会被读取的核心组件：
- `config.json`
- `generation_config.json`
- tokenizer 组件：`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`
- 权重索引与分片（HF）：
- `model.safetensors.index.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- 同目录还存在 `.bin` 分片：
- `pytorch_model.bin.index.json`
- `pytorch_model-00001-of-00002.bin`
- `pytorch_model-00002-of-00002.bin`

核验补充：
- safetensors 总大小：`13476835328` bytes（2 shard）。
- `.bin` 总大小：`13476839424` bytes（2 shard）。

### B. MiniGPT 阶段权重（项目内）
路径：`/home/hdliu/MiniGPT-4/checkpoint_stage3.pth`

作用：
- `minigpt_v2.from_config` 会加载该权重：`minigpt4/models/minigpt_v2.py:133-137`。

核验补充：
- 文件存在，大小 `679808493` bytes。

### C. 视觉编码器权重（EVA）
代码来源：
- `minigpt4/models/eva_vit.py:429-433` 固定 URL 下载 `eva_vit_g.pth` 后读取。

本机可见缓存：
- `/raid/hdliu_raid/.cache/torch/hub/checkpoints/eva_vit_g.pth`（存在，`2025249237` bytes）。

## 3.2 数据输入（读取）

### A. 样本清单 JSON
路径（映射后）：`/raid/hdliu_raid/datasets/llava/glob_json/cap_videos_14k_frame.json`

核验结果：
- 样本数：`14280`
- key 集合：`['conversations', 'id', 'ignore', 'image']`
- `image_only`: `14280`
- `video_only`: `0`
- `image+video`: `0`
- 缺失图片：`0`

### B. 实际媒体根目录
路径（映射后）：`/raid/hdliu_raid/datasets/cap_vid`

真实被引用子路径：
- `/raid/hdliu_raid/datasets/cap_vid/cap_videos_14k_frames/**/*.jpg`

核验结果：
- JSON 引用图片：`14280`
- `cap_videos_14k_frames` 下总文件：`14282`
- 未被本次 JSON 使用的 2 个文件：
- `cap_videos_14k_frames/ov-full-annotation-14k.json`
- `cap_videos_14k_frames/ov-prop-direct-14k.json`
- `cap_vid` 下其它目录（如 `cap_vid_1214_train`、`cap_vid_1214_test` 等）不参与本次训练。

## 3.3 输出与恢复路径（读写）
路径：`/raid/hdliu_raid/models/MiniGPT/checkpoints`

行为：
- 运行时创建：`/raid/hdliu_raid/models/MiniGPT/checkpoints/<job_id>/`
- 写入：
- `log.txt`
- `checkpoint_<epoch>.pth`（每 epoch）
- `result/` 子目录

关键说明：
- 当前配置 `resume_ckpt_path: null`，默认不从旧 checkpoint 恢复。

## 4. 运行环境相关的关键点
- `finetune.sh` 使用相对路径（`python train.py --cfg-path train_configs/...`），因此需要在 `MiniGPT-4` 根目录执行，或把脚本改为绝对路径。
- `start_docker.sh` 只负责起容器与挂载，不会自动 `cd /home/hdliu/MiniGPT-4`。
- `start_docker.sh:4` 仅挂载 `/raid/${USER}` 到容器内同路径；你提供的“`/raid/hdliu == /raid/hdliu_raid`”等价关系属于你的环境约定，本文已按该约定做路径核验。
- 我这边没有直接进你 Docker 里跑训练（宿主 Python 环境缺少 `ipdb/numpy/timm`），本文结论来自静态追踪 + 路径存在性核验。

## 5. 最小集合总结（你真正要盯住的）
- `~/MiniGPT-4` 最小主干：
- `finetune.sh`
- `train.py`
- `train_configs/minigptv2_finetune.yaml`
- `minigpt4/common/config.py`
- `minigpt4/configs/models/minigpt_v2.yaml`
- `minigpt4/configs/datasets/llava/capvid_frame.yaml`
- `minigpt4/tasks/base_task.py`
- `minigpt4/tasks/image_text_pretrain.py`
- `minigpt4/datasets/builders/image_text_pair_builder.py`
- `minigpt4/datasets/datasets/llava_dataset.py`
- `minigpt4/processors/blip_processors.py`
- `minigpt4/models/minigpt_v2.py`
- `minigpt4/models/minigpt_base.py`
- `minigpt4/models/base_model.py`
- `minigpt4/models/modeling_llama.py`
- `minigpt4/models/eva_vit.py`
- `minigpt4/runners/runner_base.py`

- `/raid/hdliu_raid` 最小主干：
- `/raid/hdliu_raid/models/MiniGPT/Llama-2-7b-chat-hf`
- `/raid/hdliu_raid/datasets/llava/glob_json/cap_videos_14k_frame.json`
- `/raid/hdliu_raid/datasets/cap_vid/cap_videos_14k_frames/**/*.jpg`
- `/raid/hdliu_raid/models/MiniGPT/checkpoints`（输出）
- `/raid/hdliu_raid/.cache/torch/hub/checkpoints/eva_vit_g.pth`（若已缓存）


## 6. `eval_model.sh` 真实依赖清单（静态追踪 + 路径核验）

### 6.1 最重要结论
- 实际入口链是：`eval_scripts/eval_model.sh` -> `eval_scripts/eval_vqa.py` -> `minigpt4/common/eval_utils.py::init_model` -> `minigpt_v2` 模型初始化 -> `cap_decom` 分支推理。
- `eval_model.sh` 传的是 `--dataset cap_decom`，因此命中 `eval_vqa.py` 里的 `if 'cap_decom' in args.dataset` 分支。
- 该分支的数据读取器是 `CapVidData`：读取一个 annotation JSON（字典），把 key 从 `*.mp4` 映射为同名 `*.jpg`，在 `img_path` 下开图。
- 模型加载依赖训练后权重 `checkpoint_49.pth`（配置指定）+ Llama2 基座 + EVA 视觉塔权重缓存/下载。
- 你要求“只考虑 `/raid/hdliu_raid/datasets/cap_vid`”。按这个约束，当前 YAML 中 `capvid` 默认路径（`something-something`）不符合且实测不存在，需要切到 `cap_vid` 下可用标注/图片对。

### 6.2 `~/MiniGPT-4` 下真正涉及的文件和脚本

### 6.2.1 直接被命令行引用（硬依赖）
- `eval_scripts/eval_model.sh`
- `eval_scripts/eval_vqa.py`
- `eval_configs/minigptv2_benchmark_evaluation.yaml`

对应证据：
- `eval_scripts/eval_model.sh:4-7`。

### 6.2.2 `cap_decom` 评估主链（本次参数会命中）
- 参数/配置/模型初始化：
- `minigpt4/common/eval_utils.py`
- `minigpt4/common/config.py`

- 模型与处理器主链：
- `minigpt4/models/minigpt_v2.py`
- `minigpt4/models/minigpt_base.py`
- `minigpt4/models/base_model.py`
- `minigpt4/models/modeling_llama.py`
- `minigpt4/models/eva_vit.py`
- `minigpt4/processors/blip_processors.py`

- 数据与对话模板：
- `minigpt4/datasets/datasets/vqa_datasets.py`（`CapVidData`）
- `minigpt4/conversation/conversation.py`（`CONV_VISION_minigptv2`）

关键分支证据：
- `eval_scripts/eval_vqa.py:33-36`（解析参数并构建 `Config`）。
- `eval_scripts/eval_vqa.py:153-160`（`cap_decom` 读取 `capvid.eval_file_path/img_path` 并构建 `CapVidData`）。
- `eval_scripts/eval_vqa.py:167-203`（decompose -> proposition -> conditional summary 推理链）。
- `eval_scripts/eval_vqa.py:208-209`（输出写到固定文件 `/home/hdliu/MiniGPT-4/minigptv2-14k-50e-decom-some.json`）。
- `minigpt4/common/eval_utils.py:47-63`（`init_model`：`model_cls.from_config(...)` + `vis_processor` 初始化）。
- `minigpt4/datasets/datasets/vqa_datasets.py:46-63`（`CapVidData` 的 key->jpg 映射与开图逻辑）。
- `eval_configs/minigptv2_benchmark_evaluation.yaml:8-11`（Llama2 与评估 checkpoint 路径）。

### 6.2.3 启动阶段 import 触发的附加依赖（不一定进入业务分支）
- `eval_vqa.py` 顶层会导入但 `cap_decom` 不实际使用的模块：
- `minigpt4.common.vqa_tools.VQA.*`（仅 OKVQA 分支用）
- `datasets.load_dataset`（仅 VSR 分支用）
- `OKVQAEvalData/VizWizEvalData/IconQAEvalData/GQAEvalData/VSREvalData/HMEvalData`（`cap_decom` 不用）

- `eval_utils.py` 顶层 wildcard 注册导入会触发的大量模块：
- `from minigpt4.datasets.builders import *`
- `from minigpt4.models import *`
- `from minigpt4.processors import *`
- `from minigpt4.runners import *`
- `from minigpt4.tasks import *`

说明：
- 即使你只跑 `cap_decom`，上述顶层 import 仍会在启动阶段被解释器加载；但 `cap_decom` 真正执行的数据读取是 `CapVidData`。

### 6.2.4 与当前 `cap_vid` 约束直接相关的配置问题
- 你当前脚本指定的 YAML 中：
- `evaluation_datasets.capvid.eval_file_path` = `/raid/hdliu_raid/datasets/something-something/some_some_gp_anno.json`
- `evaluation_datasets.capvid.img_path` = `/raid/hdliu_raid/datasets/something-something/some_some_frames`
- 这两条路径实测不存在，且不在 `cap_vid` 下。

对应证据：
- `eval_configs/minigptv2_benchmark_evaluation.yaml:55-56`。

### 6.3 `/raid/hdliu_raid` 下真实涉及的数据集与模型组件（仅 `cap_vid` 口径）

### 6.3.1 模型输入（读取）
- Llama2 基座目录：
- `/raid/hdliu_raid/models/MiniGPT/Llama-2-7b-chat-hf`

- 当前评估配置 checkpoint：
- `/raid/hdliu_raid/models/MiniGPT/checkpoints/cap_vid_14k_50e/checkpoint_49.pth`
- 已核验存在（`679795693` bytes）。

- EVA 视觉权重缓存（若已缓存会被直接读取，否则走 URL 下载缓存）：
- `/raid/hdliu_raid/.cache/torch/hub/checkpoints/eva_vit_g.pth`

### 6.3.2 数据输入（读取，限定 `cap_vid`）
`CapVidData` 需要：
- `eval_file_path`：JSON 字典（key 为 `...mp4` 风格 id）
- `img_path`：与 key 对应的 jpg 根目录
- 映射规则：`key.rsplit('.', 1)[0] + '.jpg'`

在 `/raid/hdliu_raid/datasets/cap_vid` 下可直接匹配该规则的主标注集合：
- `/raid/hdliu_raid/datasets/cap_vid/cap_vid_1214_test_frames/test_anno_1214.json`（`1117` 样本，缺失图片 `0`）
- `/raid/hdliu_raid/datasets/cap_vid/test_heldout_18-1_frames/test_heldout_18-1-anno.json`（`2204` 样本，缺失图片 `0`）
- `/raid/hdliu_raid/datasets/cap_vid/cap_videos_14k_frames/ov-full-annotation-14k.json`（`4640` 样本，缺失图片 `0`）

配套图片根目录分别为：
- `/raid/hdliu_raid/datasets/cap_vid/cap_vid_1214_test_frames`
- `/raid/hdliu_raid/datasets/cap_vid/test_heldout_18-1_frames`
- `/raid/hdliu_raid/datasets/cap_vid/cap_videos_14k_frames`

补充：
- 上述三份标注的 value 均为字典；`cap_decom` 分支里 `texts` 变量未用于生成 prompt（仅图片参与推理）。

### 6.3.3 输出路径（读写）
- `cap_decom` 分支输出写死为：
- `/home/hdliu/MiniGPT-4/minigptv2-14k-50e-decom-some.json`

- `run.save_path`（`/raid/hdliu_raid/models/MiniGPT/eval_res`）在 `cap_decom` 分支未被使用。

### 6.4 运行环境相关关键点
- `eval_model.sh` 调的是 `torchrun ... eval_vqa.py`（相对脚本名），因此需要在 `eval_scripts` 目录运行，或把 `eval_vqa.py` 改成绝对路径。
- 当前 `capvid` 配置默认是 `something-something` 且路径缺失；若只评估 `cap_vid`，需把 YAML 里的 `capvid.eval_file_path/img_path` 改到 `cap_vid` 下有效路径对（例如 YAML 注释里给出的 `cap_vid_1214_test_frames` 对）。
- 我没有在你环境里实际启动评估进程；结论来自静态调用链 + 路径/样本存在性核验。

### 6.5 最小集合总结（`eval_model.sh` + `cap_decom` + `cap_vid`）
- `~/MiniGPT-4` 最小代码主干：
- `eval_scripts/eval_model.sh`
- `eval_scripts/eval_vqa.py`
- `eval_configs/minigptv2_benchmark_evaluation.yaml`
- `minigpt4/common/eval_utils.py`
- `minigpt4/common/config.py`
- `minigpt4/datasets/datasets/vqa_datasets.py`
- `minigpt4/conversation/conversation.py`
- `minigpt4/models/minigpt_v2.py`
- `minigpt4/models/minigpt_base.py`
- `minigpt4/models/base_model.py`
- `minigpt4/models/modeling_llama.py`
- `minigpt4/models/eva_vit.py`
- `minigpt4/processors/blip_processors.py`

- `/raid/hdliu_raid` 最小数据/模型主干（cap_vid 口径）：
- `/raid/hdliu_raid/models/MiniGPT/Llama-2-7b-chat-hf`
- `/raid/hdliu_raid/models/MiniGPT/checkpoints/cap_vid_14k_50e/checkpoint_49.pth`
- `/raid/hdliu_raid/datasets/cap_vid/cap_vid_1214_test_frames/test_anno_1214.json`
- `/raid/hdliu_raid/datasets/cap_vid/cap_vid_1214_test_frames/**/*.jpg`
- `/raid/hdliu_raid/datasets/cap_vid/test_heldout_18-1_frames/test_heldout_18-1-anno.json`
- `/raid/hdliu_raid/datasets/cap_vid/test_heldout_18-1_frames/**/*.jpg`
- `/raid/hdliu_raid/datasets/cap_vid/cap_videos_14k_frames/ov-full-annotation-14k.json`
- `/raid/hdliu_raid/datasets/cap_vid/cap_videos_14k_frames/**/*.jpg`
