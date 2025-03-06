from openai import OpenAI
import json
import os
import ipdb
import base64
import time
from tqdm import tqdm

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

def generate(test_root, img, prompt):
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
    
    return response

test_root = "path to frames root"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

store = {}
img_lst = get_action_image_combinations(test_root)
for img in tqdm(img_lst):
    store[img] = {}
    store[img]['decom'] = generate(test_root, img, DEC)
    store[img]['proposition'] = generate(test_root, img, PROP)

    if store[img]['decom'].startswith('No'):
        store[img]['question'] = generate(test_root, img, SQ.format(store[img]['proposition']))
        store[img]['answer'] = generate(test_root, img, SA.format(store[img]['question']))
        store[img]['summary'] = generate(test_root, img, SU.format(store[img]['answer']))

    elif store[img]['decom'].startswith('Yes'):
        store[img]['summary'] = generate(test_root, img, SU_DIR.format(store[img]['proposition']))

    else:
        print("Error: ", store[img]['decom'])
        ipdb.set_trace()

    print(store[img])

with open("./gpt-4o-HAR-decom-heldin.json", "w") as f:
    json.dump(store, f)