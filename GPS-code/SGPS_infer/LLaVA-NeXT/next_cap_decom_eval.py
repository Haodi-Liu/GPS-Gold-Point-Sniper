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

DEC = "You are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and summarizing key points in order to successfully complete task of human activity recognition. Please carefully evaluate the given frames and decide if the human action recognition can be confidently completed via direct answering. If human action recognition can be directly completed, please answer \"Yes.\" If deeper visual information capturing and more complex reasoning structures are required to reliably complete human action recognition, please answer \"No.\""
PROP = "Please focus on the visual details and temporal dynamics relevant to the ongoing human action and summarize 5-7 key propositions(each no more than 20 words) essential for correctly completing the task of human action recognition. These propositions are visually grounded and necessary for successful task completion. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Reasonable and visually grounded reasonings are allowed during the generations of propositions. Also be sure to leverage the temporal changes in the frames. Please give your answer in format of a few bullet points like \"1. ... 2. ... 3. ...\n"
SQ = "Treating the propositions {} as guidelines, please propose 3-5 questions(each no more than 20 words) that would seek for more information in order to recognize the ongoing human action more accurately. The questions asked have to be related to the ongoing human action and can be answered with visually grounded evidences from the given video frames. Also be sure to leverage the temporal changes in the frames and never make any ungrounded assumptions in the questions."
SQ1 = "Please propose 3-5 questions(each no more than 20 words) that would seek for more information in order to recognize the ongoing human action more accurately. The questions asked have to be related to the ongoing human action and can be answered with visually grounded evidences from the given video frames. Also be sure to leverage the temporal changes in the frames and never make any ungrounded assumptions in the questions."
SA = "Based on the given video frames, please concisely answer each of the previously proposed questions {} with each anwer contains no more than 30 words. Please leverage the fine-grained visual details and temporal dynamics related to human action recognition from the given video frames to answer the questions. Make sure every part of your answers is visually grounded(either directly observed or reliably reasoned) and never make any random guesses so as to avoid hallucinations."
SU = "Based on the previously obtained answers {}, please offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."
SU_DIR = "Based on the previously obtained propositions {}, please offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."
SU_DIR1 = "Please offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."

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

def generate(qs, conv_template):

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

    return text_outputs[0]

checkpoint = "path to llava-next models checkpoint"
#model_name = "llava_llama3"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(checkpoint, None, model_name, load_8bit=False, load_4bit=True, device_map=device_map, attn_implementation="flash_attention_2") # Add any other thing you want to pass in llava_model_args

model.eval()

test_root = "path to frames root"
store = {}
img_lst = get_action_image_combinations(test_root)
for img in tqdm(img_lst):
    store[img] = {}
    image = Image.open(os.path.join(test_root, img))
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    #conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
    conv_template = "qwen_2"

    store[img]['decom'] = generate(DEC, conv_template)
    
    store[img]['proposition'] = generate(PROP, conv_template)
    if store[img]['decom'].startswith('No'):
        store[img]['question'] = generate(SQ.format(store[img]['proposition']), conv_template)
        store[img]['answer'] = generate(SA.format(store[img]['question']), conv_template)
        store[img]['summary'] = generate(SU.format(store[img]['answer']), conv_template)

    elif store[img]['decom'].startswith('Yes'):
        store[img]['summary'] = generate(SU_DIR.format(store[img]['proposition']), conv_template)
    
    else:
        print("Error: ", store[img]['decom'])

with open("./{}-decom-perf-heldout.json".format(checkpoint.rsplit('/')[-1]), "w") as f:
    json.dump(store, f)