import ipdb
import json
import time
from llamaapi import LlamaAPI
import json
import numpy as np
import re
import warnings
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

TEMP_EVAL_USER = "Your current role is an evaluator for human action recognition tasks. Your task is to determine the relationship \
between a given summary description : '{}' and each of the propositions: '{}' as entailment, neutrality, or contradiction. The propositions, as evaluation standards, are guaranteed to be true. \
While comparing the summary against a proposition, please stick with the literal meaning and never make any ungrounded inference. \
Some semantically similar actions and object types can be considered as having the same meaning. \
The summary entails a proposition if and only if content of the summary is related to the proposition and potentially helps to support the proposition's validity. \
The summary contradicts a proposition if and only if the main content of the summary is related to the proposition but is clearly contrary to the meaning of the proposition. \
The relation is neural if the summary's content has no tendency to support or refute the proposition as it might talk about something uncrucial to the validity of proposition. \
Please organize your output according to the following format: \
1. [Relation to Proposition 1 (please answer with the number 0 or 1 or 2)], [A one-sentence explanation of your reasoning]; \
2. [Relation to Proposition 2 (please answer with the number 0 or 1 or 2)], [A one-sentence explanation of your reasoning]; \
    ...and so on until all propositions are covered. \
Here, the relationship to each proposition is represented as: \
0 for Contradiction, 1 for Neutrality, 2 for Entailment. Please strictly follow the format above and be sure to embrace the relation score with []. No need to add any additional content."

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
checkpoint = "llava-onevision-qwen2-7b-ov-14k"

# Initialize the SDK
llama = LlamaAPI("API-Key")

#with open(os.path.join(root_dir, 'test_anno_1214.json'), 'r', encoding='utf-8') as file:
with open(os.path.join(root_dir, 'test_heldout_18-1-anno.json'), 'r', encoding='utf-8') as file:
    ann_data = json.load(file)

with open(os.path.join(root_dir, '{}-decom-perf-heldout.json'.format(checkpoint)), 'r', encoding='utf-8') as file:
#with open(os.path.join(root_dir, '{}-eval-demo-heldout.json'.format(checkpoint)), 'r', encoding='utf-8') as file:
    res_data = json.load(file)

img_paths = dict()
for img in ann_data:
    img_paths[img.rsplit('.')[0] + '.jpg'] = img

scores = []
store = {}

for index, path in enumerate(tqdm(res_data, desc="Processing images")):

    label = path.split('/')[0]
    if label in ACT_COR:
        label = ACT_COR[label]
    store[path] = {}
    
    props = ann_data[img_paths[path]]['proposition']
    index = int(props.split('\n')[-1].split('.')[0]) + 1

    add_prop = "\n{}. The ongoing human action is {}.".format(index, label)
    props = props + add_prop

    api_request_json = {
    "model": "llama3.3-70b",
    "messages": [
        {"role": "user", "content": TEMP_EVAL_USER.format(res_data[path]['summary'], props)},
        {"role": "assistant", "content": "Your evaluation result is: "},
        ],
    "max_token" : 1024,
    "temperature" : 0.1,
    "top_p": 1.0,
    "frequency_penalty":1.0,
    }

    while True:
        try:
            response = llama.run(api_request_json)
            if isinstance(response.json(), dict):
                break
        
        except Exception as e:
            print(e)
        time.sleep(0.5)

    try:
        text_output = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        with open("./{}-backup.json".format(checkpoint), "w") as f:
            json.dump(store, f)
        ipdb.set_trace()

    print(text_output)
    
    # 使用正则表达式匹配方括号内的整数
    matches = re.findall(r'\[(\d+)\]', text_output)

    # 将匹配到的字符串转换为整数
    try:
        relations = np.array([int(match) for match in matches])
    except Exception as e:
        print(f"An error occurred: {e}")
        with open("./{}-backup.json".format(checkpoint), "w") as f:
            json.dump(store, f)
        ipdb.set_trace()

    num_con, num_neu, num_ent = (relations == 0).sum(), (relations == 1).sum(), (relations == 2).sum()
    score = (num_ent - num_con) / len(relations)
    scores.append(score)
    store[path]['summary'] = res_data[path]['summary']
    store[path]['proposition'] = props
    store[path]['eval_res'] = text_output
    store[path]['relations'] = relations.tolist()
    store[path]['score'] = score

print(f"Average score of evaluation: {np.mean(scores)}")
print(checkpoint, "decom-heldout")
#with open("./{}-eval-sum-prop-heldout.json".format(checkpoint), "w") as f:
with open("./{}-sum-prop-decom-heldout.json".format(checkpoint), "w") as f:
    json.dump(store, f)