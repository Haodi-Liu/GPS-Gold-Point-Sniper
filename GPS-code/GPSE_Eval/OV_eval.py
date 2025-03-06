from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import ipdb
from PIL import Image
import random
import json
import copy
import torch
import numpy as np
import re
import os
from tqdm import tqdm
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")

TEMP_HALL = "Your current task is to evaluate the degree of alignment in the given summary description {} with respect to the visual details in the provided video frames {}. \
Hallucination is defined as elements in the summary description that are either not present in the video frames or contradict the content of the video frames. \
Please rate the degree of hallucination in the summary description on a scale from 0 to 1, where a higher score indicates a higher degree of hallucination. \
When evaluating hallucination, please objectively assess the authenticity of the summary description (without assuming it to be true or false) and strictly adhere to the \
visual details and temporal dynamics of the video frames. Additionally, please evaluate the summary description’s coverage of the visual details and temporal dynamics in the video frames, \
especially those critical for recognizing human actions. Rate the degree of detail coverage on a scale from 0 to 1, where a higher score indicates more comprehensive and detailed coverage and understanding of the visual details. \
Your output format should be: \
Hallucination score (a floating-point number between 0 and 1), followed by your reasoning for the score. \
Detail coverage score (a floating-point number between 0 and 1), followed by your reasoning for the score."

ACT_COR = {"person_braids_hair_of_person":"person_styles_hair_of_person", "person_grabs_person_by_forearm":"person_grabs_person_by_arm", \
           "person_interacts_with_tablet":"person_interacts_with_mobile_device", "person_paints_fingernails":"person_applies_something_to_fingers",\
            "person_pours_coffee_into_mug":"person_pours_drinks_into_mug", "person_pulls_wheeled_trashcan":"person_pulls_container", "person_sets_upright_glass":"person_sets_upright_cup",\
            "person_ties_jacket_around_waist":"person_puts_on_jacket"}

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

root_dir = "path to frames root"
pretrained = "path to OV72B"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
conv_template = "qwen_1_5"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, load_8bit=False, load_4bit=True, device_map=device_map, attn_implementation="flash_attention_2")  # Add any other thing you want to pass in llava_model_args
model.eval()
with open(os.path.join(root_dir, 'llama3-llava-next-8b-14k-noprop-perf-heldout.json'), 'r', encoding='utf-8') as file:
    res_data = json.load(file)
name = "llama3-llava-next-8b-14k-noprop-heldout"
#with open(os.path.join(root_dir, 'test_anno_1214.json'), 'r', encoding='utf-8') as file:
with open(os.path.join(root_dir, 'test_heldout_18-1-anno.json'), 'r', encoding='utf-8') as file:
    ann_data = json.load(file)


img_paths = dict()
for img in ann_data:
    img_paths[img.rsplit('.')[0] + '.jpg'] = img

# images = []
# images.append(Image.open(os.path.join(root_dir, path1)))
# images.append(Image.open(os.path.join(root_dir, path2)))
# image_sizes = [image.size for image in images]
# image_tensor = process_images(images, image_processor, model.config)
hall_scores = []
cov_scores = []
store = {}
for path in tqdm(res_data, desc="Processing images"):
    #if path not in already:
    label = path.split('/')[0]
    if label in ACT_COR:
        label = ACT_COR[label]
    store[path] = {}
    image = Image.open(os.path.join(root_dir, path))
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    image_sizes = [image.size]
    
    question = TEMP_HALL.format(res_data[path]['summary'], DEFAULT_IMAGE_TOKEN)
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    try:
        # hall_score = float(text_outputs[0].split('\n')[0].rsplit(' ')[-1])
        # cov_score = float(text_outputs[0].split('\n')[-2].rsplit(' ')[-1])
        hall_score = float(text_outputs[0].split('score: ')[1].split('\n')[0])
        cov_score = float(text_outputs[0].split('score: ')[-1].split('\n')[0])
    except Exception as e:
        print(f"An error occurred: {e}")
        ipdb.set_trace()

    # hall_reason = text_outputs[0].split('\n')[1]
    # cov_reason = text_outputs[0].split('\n')[-1]
    hall_reason = text_outputs[0].split('score: ')[1].split('\n')[1]
    cov_reason = text_outputs[0].split('score: ')[-1].split('\n')[1]
    hall_scores.append(hall_score)
    cov_scores.append(cov_score)
    store[path]['prompt'] = prompt_question
    store[path]['hall_score'] = hall_score
    store[path]['hall_reason'] = hall_reason
    store[path]['cov_score'] = cov_score
    store[path]['cov_reason'] = cov_reason
    print(store[path])

print(f"Average score of hallucination: {np.mean(hall_scores)}")
print(f"Average score of detail coverage: {np.mean(cov_scores)}")
print(name)
with open("./llama3-llava-next-8b-14k-noprop-vision-eval-heldout.json", "w") as f:
    json.dump(store, f)