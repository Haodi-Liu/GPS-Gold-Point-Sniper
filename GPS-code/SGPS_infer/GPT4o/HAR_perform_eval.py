from openai import OpenAI
import json
import os
import ipdb
import base64
import time
from tqdm import tqdm

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

prompt = "Please offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."

test_root = "path to frames root"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

store = {}
img_lst = get_action_image_combinations(test_root)
for img in tqdm(img_lst):
    base64_img = encode_image(os.path.join(test_root, img))

    while True:
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and summarizing key points in order to successfully complete tasks involving complex reasoning."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
                ],
            )
            break

        except Exception as e:
            print(e)
        time.sleep(0.5)

    try:
        response = chat_completion.choices[0].message.content

    except Exception as e:
        print(e)
        with open("./gpt-4o-backup.json", "w") as f:
                json.dump(store, f)
        ipdb.set_trace()

    store[img] = response
    print(store[img])

with open("./gpt-4o-HAR-heldout.json", "w") as f:
    json.dump(store, f)