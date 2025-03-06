from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM
import ipdb
import os
import json
from tqdm import tqdm
torch.manual_seed(1234)

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
checkpoint = "path to trained Qwen-VL-Chat checkpoint"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="cuda").eval()
#model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("path to Qwen-VL-Chat", trust_remote_code=True)

test_root= "path to test frames"
store = {}
img_lst = get_action_image_combinations(test_root)
for img in tqdm(img_lst):
    query = tokenizer.from_list_format([
        {'image': os.path.join(test_root, img)},
        {'text': 'Please describe the ongoing human action within the given video frames and be sure state the recognized action(including the interacted objects if possible) with one sentence at last.'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    store[img] = response

with open("./{}-direct-perf-heldout.json".format(checkpoint.rsplit('/')[-1]), "w") as f:
    json.dump(store, f)