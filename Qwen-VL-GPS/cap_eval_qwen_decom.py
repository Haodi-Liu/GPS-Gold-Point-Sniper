from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM
import ipdb
import os
import json
from tqdm import tqdm
torch.manual_seed(1234)

DEC = "You are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and summarizing key points in order to successfully complete task of human activity recognition. Please carefully evaluate the given frames and decide if the human action recognition can be confidently completed via direct answering. If human action recognition can be directly completed, please answer \"Yes.\" If deeper visual information capturing and more complex reasoning structures are required to reliably complete human action recognition, please answer \"No.\""
PROP = "Please focus on the visual details and temporal dynamics relevant to the ongoing human action and summarize 5-7 key propositions(each no more than 20 words) essential for correctly completing the task of human action recognition. These propositions are visually grounded and necessary for successful task completion. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Reasonable and visually grounded reasonings are allowed during the generations of propositions. Also be sure to leverage the temporal changes in the frames. Please give your answer in format of a few bullet points like \"1. ... 2. ... 3. ...\n"
SQ = "Treating the propositions {} as guidelines, please propose 3-5 questions(each no more than 20 words) that would seek for more information in order to recognize the ongoing human action more accurately. The questions asked have to be related to the ongoing human action and can be answered with visually grounded evidences from the given video frames. Also be sure to leverage the temporal changes in the frames and never make any ungrounded assumptions in the questions."
SA = "Based on the given video frames, please concisely answer each of the previously proposed questions {} with each anwer contains no more than 30 words. Please leverage the fine-grained visual details and temporal dynamics related to human action recognition from the given video frames to answer the questions. Make sure every part of your answers is visually grounded(either directly observed or reliably reasoned) and never make any random guesses so as to avoid hallucinations."
SU = "Based on the previously obtained answers {}, please offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."
SU_DIR = "Based on the previously obtained propositions {}, please offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."

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

checkpoint = "/raid/hdliu_raid/models/Qwen/output_qwen/Qwen-VL-Chat-14k-5epoch"
#checkpoint = "/raid/hdliu/models/Qwen/Qwen-VL-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
#model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto").eval()
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="cuda").eval()

#model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("/raid/hdliu_raid/models/Qwen/Qwen-VL-Chat", trust_remote_code=True)

#test_root = "/raid/hdliu/datasets/cap_vid/cap_vid_1214_test_frames"
#test_root = "/raid/hdliu_raid/datasets/haa500_v1_1/haa500_test_frames"
test_root = "/raid/hdliu_raid/datasets/something-something/some_some_frames"
store = {}
img_lst = get_action_image_combinations(test_root)
for img in tqdm(img_lst):
    store[img] = {}
    query = tokenizer.from_list_format([
        {'image': os.path.join(test_root, img)},
        {'text': DEC},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    store[img]['decom'] = response

    query = tokenizer.from_list_format([
        {'image': os.path.join(test_root, img)},
        {'text': PROP},
    ])

    response, history = model.chat(tokenizer, query=query, history=history)
    store[img]['proposition'] = response

    if store[img]['decom'].startswith('No'):

        query = tokenizer.from_list_format([
            {'image': os.path.join(test_root, img)},
            {'text': SQ.format(store[img]['proposition'])},
        ])

        response, history = model.chat(tokenizer, query=query, history=history)
        store[img]['question'] = response

        query = tokenizer.from_list_format([
            {'image': os.path.join(test_root, img)},
            {'text': SA.format(store[img]['question'])},
        ])

        response, history = model.chat(tokenizer, query=query, history=history)
        store[img]['answer'] = response

        query = tokenizer.from_list_format([
            {'image': os.path.join(test_root, img)},
            {'text': SU.format(store[img]['answer'])},
        ])

        response, history = model.chat(tokenizer, query=query, history=history)
        store[img]['summary'] = response

    elif store[img]['decom'].startswith('Yes'):

        query = tokenizer.from_list_format([
            {'image': os.path.join(test_root, img)},
            {'text': SU_DIR.format(store[img]['proposition'])},
        ])

        response, history = model.chat(tokenizer, query=query, history=history)
        store[img]['summary'] = response

    else:
        print("Error: ", store[img]['decom'])
        ipdb.set_trace()


with open("/home/hdliu/Qwen-VL/{}-decom-perf-some.json".format(checkpoint.rsplit('/')[-1]), "w") as f:
#with open("/home/hdliu/Qwen-VL/{}-decom-perf-heldin.json".format(checkpoint.rsplit('/')[-1]), "w") as f:
    json.dump(store, f)