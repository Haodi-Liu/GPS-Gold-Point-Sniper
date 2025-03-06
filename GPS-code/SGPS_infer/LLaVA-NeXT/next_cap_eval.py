from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import os
import json
from tqdm import tqdm
from PIL import Image
import ipdb
import copy
import torch
import warnings

warnings.filterwarnings("ignore")

def get_action_image_combinations(directory):
    action_image_combinations = []
    
    # 遍历目录中的所有文件和文件夹
    for action_folder in os.listdir(directory):
        action_folder_path = os.path.join(directory, action_folder)
        
        # 检查是否是文件夹
        if os.path.isdir(action_folder_path):
            for image_file in os.listdir(action_folder_path):
                image_file_path = os.path.join(action_folder_path, image_file)
                
                # 检查是否是文件
                if os.path.isfile(image_file_path):
                    action_image_combinations.append(f"{action_folder}/{image_file}")
    
    return action_image_combinations

checkpoint = "path to llava-next models checkpoint"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(checkpoint, None, model_name, load_8bit=False, load_4bit=True, device_map=device_map, attn_implementation="flash_attention_2") # Add any other thing you want to pass in llava_model_args

model.eval()
test_root = "path to frames root"

store = {}
img_lst = get_action_image_combinations(test_root)
for img in tqdm(img_lst):

    image = Image.open(os.path.join(test_root, img))
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_2" # Make sure you use correct chat template for different models
    qs = "\nPlease offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."
    question = DEFAULT_IMAGE_TOKEN + qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    store[img] = text_outputs[0]
    print(text_outputs)

with open("./{}-direct-perf-heldin.json".format(checkpoint.rsplit('/')[-1]), "w") as f:
    json.dump(store, f)