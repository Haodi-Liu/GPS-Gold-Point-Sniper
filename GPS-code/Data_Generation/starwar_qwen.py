import json
import uuid
import ipdb
import os
import glob

PROP_TEMPLATE_IMG = "Please focus on the visual details relevant to the ongoing human action and summarize the key propositions essential for correctly completing the task of human action recognition. These propositions are visually grounded and necessary for successful task completion. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Reasonable and visually grounded reasonings are allowed during the generations of propositions. Please give your answer in format of a few bullet points like \"1. ... 2. ... 3. ...\n"

PROP_TEMPLATE_FRAME = "Please focus on the visual details and temporal dynamics relevant to the ongoing human action and summarize the key propositions essential for correctly completing the task of human action recognition. These propositions are visually grounded and necessary for successful task completion. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Reasonable and visually grounded reasonings are allowed during the generations of propositions. Also be sure to leverage the temporal changes in the frames. Please give your answer in format of a few bullet points like \"1. ... 2. ... 3. ...\n"

DECOM_DECI_IMG = "You are an excellent visual language assistant highly skilled in observing and analyzing an given image, capturing fine-grained visual details and summarizing key points in order to successfully complete tasks involving complex reasoning. We now aim to accomplish a task of human activity recognition based on the given image <image>. Please carefully evaluate the given image and decide if the human action recognition can be confidently completed via direct answering. If human action recognition can be directly completed, please answer \"Yes.\" If deeper visual information capturing and more complex reasoning structures are required to reliably complete human action recognition, please answer \"No.\""

DECOM_DECI_FRAME = "You are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and summarizing key points in order to successfully complete tasks involving complex reasoning. We now aim to accomplish a task of human activity recognition based on the given video frames <img>{}</img>. Please carefully evaluate the given frames and decide if the human action recognition can be confidently completed via direct answering. If human action recognition can be directly completed, please answer \"Yes.\" If deeper visual information capturing and more complex reasoning structures are required to reliably complete human action recognition, please answer \"No.\""

SUM_DIRECT = "Based on the previously obtained propositions, please offer a visually grounded summarizing description of the ongoing human action. Reasoning supported by reliable external knowledge and visual details is permitted. Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."

SQ_FRAME = "Treating the propositions generated just now as guidelines, please propose 5-8 questions that would seek for more information in order to recognize the ongoing human action more accurately. The questions mainly serve two roles: 1.To verify and query the information in the propositions based on the video frames, ensuring its reliability. 2. To refine the reliable parts of the propositions as needed, aiming to uncover additional clues that are useful for the task. The questions asked have to be related to the ongoing human action and can be answered with visually grounded evidences from the given video frames. Also be sure to leverage the temporal changes in the frames."

SQ_IMG = "Treating the propositions generated just now as guidelines, please propose 5-8 questions that would seek for more information in order to recognize the ongoing human action more accurately. The questions mainly serve two roles: 1.To verify and query the information in the propositions based on the image, ensuring its reliability. 2. To refine the reliable parts of the propositions as needed, aiming to uncover additional clues that are useful for the task. The questions asked have to be related to the ongoing human action and can be answered with visually grounded evidences from the given image." 

SA_FRAME = "Based on the given video frames, please answer the previously proposed questions. Please leverage the fine-grained visual details and temporal dynamics related to human action recognition from the given video frames to answer the questions. Make sure every part of your answers is visually grounded(either directly observed or reliably reasoned) and never make any random guesses so as to avoid hallucinations. Also be sure to leverage the temporal changes in the frames."

SA_IMG = "Based on the given image, please answer the previously proposed questions. Please leverage the fine-grained visual details and temporal dynamics related to human action recognition from the given image to answer the questions. Make sure every part of your answers is visually grounded(either directly observed or reliably reasoned) and never make any random guesses so as to avoid hallucinations."

SUM_FULL_FRAME = "Taking into account the information within the previously obtained questions and answers, please focus on the visual details and temporal dynamics relevant to the task and offer a visually grounded summarizing description of the ongoing human action (Reasoning supported by reliable external knowledge and visual details is permitted). Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."

SUM_FULL_IMG = "Taking into account the information within the previously obtained questions and answers, please focus on the visual details relevant to the task and offer a visually grounded summarizing description of the ongoing human action (Reasoning supported by reliable external knowledge and visual details is permitted). Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to state the recognized action(including the interacted objects if possible) with one sentence at last."

json_root = "./cap_videos_7k"

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

# Function to generate a random ID
def generate_id():
    return str(uuid.uuid4())

# Read the input file
direct_answer = os.path.join(json_root, 'ov-prop-direct-novall.json')
with open(direct_answer, 'r', encoding='utf-8') as f:
    da_data = json.load(f)

full_ann = os.path.join(json_root, 'ov-full-annotation-Novall.json')
with open(full_ann, 'r', encoding='utf-8') as f:
    fa_data = json.load(f)

# lst = get_action_image_combinations('./cap_videos_14k_frames')
# set1 = set(da_data.keys())
# set2 = set(fa_data.keys())
# all_set = set1.union(set2)
# ipdb.set_trace()

# da_data.pop('person_closes_car_door/08B2A62C-D3CE-43BC-933D-9E10C7CEC965_2.mp4')
transformed_data = []
img_root = "./cap_videos_7k_frames"
for img, content in da_data.items():
    # img_base = os.path.join("cap_images/images", img.rsplit('.')[0] + '_frame_*.jpg')
    # matched_images = glob.glob(img_base)

    # # 选择第一个匹配的图像文件，如果有多个匹配，也可以根据需求进行选择
    # if matched_images:
    #     img_path = matched_images[0]
    # else:
    #     img_path = None  # 或者处理找不到匹配文件的情况

    img_path = os.path.join(img_root, img.rsplit('.')[0] + '.jpg')
 
    new_item = {
                "id": generate_id(),
                "conversations": [
                    {
                        "from": "user",
                        "value": DECOM_DECI_FRAME.format(img_path)
                    },
                    {
                        "from": "assistant",
                        "value": "Yes, the human action recognition can be directly completed."
                    },
                    {
                        "from": "user",
                        "value": PROP_TEMPLATE_FRAME
                    },
                    {
                        "from": "assistant",
                        "value": content["proposition"]
                    },
                    {
                        "from": "user",
                        "value": SUM_DIRECT
                    },
                    {
                        "from": "assistant",
                        "value": content["summary"]
                    }
                ]
            }
    
    transformed_data.append(new_item)

for img, content in fa_data.items():    
    # img_base = os.path.join("cap_images/images", img.rsplit('.')[0] + '_frame_*.jpg')
    # matched_images = glob.glob(img_base)

    # # 选择第一个匹配的图像文件，如果有多个匹配，也可以根据需求进行选择
    # if matched_images:
    #     img_path = matched_images[0]
    # else:
    #     img_path = None  # 或者处理找不到匹配文件的情况
    
    img_path = os.path.join(img_root, img.rsplit('.')[0] + '.jpg')
    new_item = {
                "id": generate_id(),
                "conversations": [
                    {
                        "from": "user",
                        "value": DECOM_DECI_FRAME.format(img_path)
                    },
                    {
                        "from": "assistant",
                        "value": "No, deeper visual information capturing and more complex reasoning structures are needed to reliably complete human action recognition."
                    },
                    {
                        "from": "user",
                        "value": PROP_TEMPLATE_FRAME
                    },
                    {
                        "from": "assistant",
                        "value": content["proposition"]
                    },
                    {
                        "from": "user",
                        "value": SQ_FRAME
                    },
                    {
                        "from": "assistant",
                        "value": content["questions"]
                    },
                    {
                        "from": "user",
                        "value": SA_FRAME
                    },
                    {
                        "from": "assistant",
                        "value": content["answers"]
                    },
                    {
                        "from": "user",
                        "value": SUM_FULL_FRAME
                    },
                    {
                        "from": "assistant",
                        "value": content["summary"]
                    }
                ]
            }
    

    transformed_data.append(new_item)

output_file = 'cap_videos_7k_frame_qwen.json'
ipdb.set_trace()
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, indent=4, ensure_ascii=False)