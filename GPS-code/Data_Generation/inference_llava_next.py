# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, tokenizer_image_token_batch, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import ipdb
from PIL import Image
import requests
import copy
import torch
import argparse
import sys
from tqdm import tqdm
import warnings
import os
import cv2
from pathlib import Path
import numpy as np
from decord import VideoReader, cpu
import json
import math
warnings.filterwarnings("ignore")

# 要点提示词，无标签，用作坏问题组生成和分解决策标注 (根据这些要点能否鲁棒地推断出标签，不能则需要分解)
PROP_TEMPLATE_WO = "\nYou are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and summarizing key points \
        in order to successfully complete the tasks such as recognizing human actions, detecting events, engaging in daily conversations, and performing complex reasoning. We now aim to accomplish \
        a task of {} based on the given video frames {}. Please focus on the visual details and temporal dynamics relevant to the ongoing human action and summarize the key propositions \
        essential for correctly completing the task, which are visually grounded and necessary for successful task completion. Please ignore the background details irrelevant to the task and never mention \
        any visual details you can't clearly capture. Reasonable and visually grounded reasonings are allowed during the generations of propositions. Also be sure to leverage the temporal changes in the frames. Please give your answer in format of a few bullet points like \"1. ... 2. ... 3. ...\n"

# 要点提示词，有标签，
PROP_TEMPLATE = "\nYou are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and summarizing key points \
            in order to successfully complete the tasks such as recognizing human actions, detecting events, engaging in daily conversations, and performing complex reasoning. We now aim to accomplish \
            a task of {} based on the given video frames {} with the ground truth label being {}. Fully believing in the ground truth label, please focus on the visual details and temporal dynamics relevant to the ongoing human activity and summarize the key propositions \
            essential for correctly completing the task, which are visually grounded and necessary for successful task completion. Please ignore the background details irrelevant to the task and never mention \
            any visual details you can't clearly capture. Try not to repeat the ground truth label, although you might have to mention some parts of it. Reasonable and visually grounded reasonings are allowed during the generations of propositions. \
            Also be sure to leverage the temporal changes in the frames. Please give your answer in format of a few bullet points like \"1. ... 2. ... 3. ...\n"

# 基于生成的要点，提出问题，作为自问部分，用以验证要点，并对可靠部分进一步细化
SQ_TEMPLATE = "You are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and utilizing external knowledge \
            in order to successfully complete the tasks such as recognizing human actions, detecting events, engaging in daily conversations, and performing complex reasoning. We now aim to accomplish \
            a task of {} based on the given video frames {}. Treating the propositions {} as guidelines, please propose 5-8 questions that would seek for more information in order to recognize the \
            ongoing human action more accurately. The questions mainly serve two roles: 1.To verify and query the information in the propositions based on the video frames, ensuring its reliability. \
            2. To refine the reliable parts of the propositions as needed, aiming to uncover additional clues that are useful for the task. The questions asked have to be related to the ongoing human action and can be answered with \
            visually grounded evidences from the given video frames. Also be sure to leverage the temporal changes in the frames."

#自答部分，对自我提出的问题进行回答，回答要有视觉根据，且允许合理的推理
SA_TEMPLATE = "Based on the given video frames {}, please answer the previously proposed questions {}. \
            Please leverage the fine-grained visual details and temporal dynamics related to the task of {} from the given video frames to answer the questions. \
            Make sure every part of your answers is visually grounded(either directly observed or reliably reasoned) and never make any \
            random guesses so as to avoid hallucinations. Also be sure to leverage the temporal changes in the frames."

# 汇总自问自答的结果，并给出最终的动作识别答案
SUM_TEMPLATE = "\nYou are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and utilizing external knowledge \
            in order to successfully complete the tasks such as recognizing human actions, detecting events, engaging in daily conversations, and performing complex reasoning. We now aim to accomplish \
            a task of {} based on the given video frames {}. Taking into account the information within the previously obtained questions {} and answers {}, please focus on the visual details and \
            temporal dynamics relevant to the task and offer a visually grounded summarizing description of the ongoing human action (Reasoning supported by reliable external knowledge and visual details is permitted). \
            Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."

# 基于无标签产生的要点决策是否对问题进行分解（经历自问自答等一系流程）
DEC_TEMPLATE = "\nYou are an excellent visual language assistant, skilled in observing and analyzing given video frames, capturing fine visual details, and summarizing key points to successfully accomplish tasks such as recognizing human actions, \
            detecting events, engaging in daily conversations, and performing complex reasoning. Our current goal is to complete the task of {} based on the given video frames {}. Now, you must make a very important decision: if you are confident that you \
            can complete the task based on directly observed visual and temporal details, please provide your final answer immediately. Otherwise, you will need to break down the task, using techniques like generating propositions, self-querying, and other \
            complex approaches to deeply explore visual information and perform complex reasoning. The ground truth label for the current video frames is {} and some propositions {} are given for reference. Based on your observation and understanding of the visual information, determine \
            if task decomposition is necessary to accurately identify the person's action in the video frames. Please provide a concise explanation of your reasoning based on the visual information and conclude with a 'yes' (task decomposition needed) or 'no' (task decomposition not needed).\
            These propositions represent your initial observations of the video frames. You must disregard the label and objectively assess whether these propositions {} sufficiently support the label {} as an inferred result for action recognition. \
            If the label can be definitively inferred based solely on the propositions, the answer is 'no'; if the propositions contradict the label or do not provide enough information to definitively infer the label, the answer is 'yes.'"

# 汇总靠谱的无标签要点（不需分解便可直接基于要点得出最终标签），并给出最终的动作识别答案
SUM_TEMPLATE_PROP = "\nYou are an excellent visual language assistant highly skilled in observing and analyzing given video frames, capturing fine-grained visual details and utilizing external knowledge \
            in order to successfully complete the tasks such as recognizing human actions, detecting events, engaging in daily conversations, and performing complex reasoning. We now aim to accomplish \
            a task of {} based on the given video frames {}. Taking into account the information within the previously obtained propositions {}, please focus on the visual details and \
            temporal dynamics relevant to the task and offer a visually grounded summarizing description of the ongoing human action (Reasoning supported by reliable external knowledge and visual details is permitted). \
            Please ignore the background details irrelevant to the task and never mention any visual details you can't clearly capture. Also be sure to leverage the temporal changes in the frames and state the recognized action(including the interacted objects if possible) with one sentence at last."

# directory 是 图像/视频 数据的目录，下由各种动作目录组成，动作目录下便是相应的图像/视频。
# 此函数返回的是目录下所有的“动作/视频图像名”组合。lst 是动作的列表，是我临时额外添加的
# 在batch size为1的情况下，将所有动作均分成多等份，多份并行跑提高效率，后期batch size大于1
# 的视频推断实现好了可以去掉
#def get_action_image_combinations(directory, lst):
def get_action_image_combinations(directory):
    """
    Get all combinations of action and image from the given directory.

    Args:   
        directory: The directory containing the action folders.

    Returns:
        A list of strings, where each string is in the format "action/image".
    """
    action_image_combinations = []
    
    # 遍历目录中的所有文件和文件夹
    for action_folder in os.listdir(directory):
        #if action_folder in lst:
        action_folder_path = os.path.join(directory, action_folder)
        
        # 检查是否是文件夹
        if os.path.isdir(action_folder_path):
            for image_file in os.listdir(action_folder_path):
                image_file_path = os.path.join(action_folder_path, image_file)
                
                # 检查是否是文件
                if os.path.isfile(image_file_path):
                    action_image_combinations.append(f"{action_folder}/{image_file}")
        
    return action_image_combinations

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", default=".", required=False)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--for_get_frames_num", type=int, default=5)
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=True)
    return parser.parse_args()


def load_video(video_path,args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()  # fps: 12
    fps = round(vr.get_avg_fps() * (3/5))
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time

# 确保每个视频都抽出同样数目的帧，便于批量处理
def load_video_same(video_path, args):
    """
    Load a fixed number of frames from the video.
    
    Args:
        video_path (str): Path to the video file.
        args (Namespace): Command-line arguments.
        num_frames (int): Number of frames to sample from the video.
        
    Returns:
        np.ndarray: Array containing the sampled frames.
        str: Comma-separated string of frame times.
        float: Total video duration in seconds.
    """
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0.0
        
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    
    # Use np.linspace to sample a fixed number of frames
    frame_idx = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time_str, video_time

def pad_and_stack(all_input_ids, pad_token_id):
    # Find the maximum length of the second dimension
    max_length = max(ids.shape[1] for ids in all_input_ids)
    
    # Pad all tensors to the maximum length along the second dimension with left padding
    padded_input_ids = [
        torch.cat([torch.full((1, max_length - ids.shape[1]), pad_token_id).cuda(), ids], dim=1)
        for ids in all_input_ids
    ]
    
    # Stack the padded tensors to create a (n, m) tensor (no need to squeeze)
    stacked_input_ids = torch.stack(padded_input_ids, dim=0)

    # Remove the first dimension (of size 1) by squeezing
    stacked_input_ids = stacked_input_ids.squeeze(1)
    
    return stacked_input_ids

class LLaVANeXTEngine:
    def __init__(self, args, pretrained="path to OV72B", 
                 model_name="llava_qwen", load_8bit=False, load_4bit=True, device_map="auto", attn_implementation="flash_attention_2"):
        self.args = args
        self.pretrained = pretrained
        self.model_name = model_name
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.tokenizer, self.model, self.image_processor, self.max_length = \
            load_pretrained_model(
                pretrained, 
                None, 
                model_name, 
                load_8bit=load_8bit,
                load_4bit=load_4bit,
                device_map=device_map, 
                attn_implementation=attn_implementation)  # Add any other thing you want to pass in llava_model_args
        self.model.eval()
        self.conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        self.question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
        self.batch_size = 2
        self.image_size = (910, 512)
        self.task = "human activity recognition"
        self.prop_temp = PROP_TEMPLATE
        self.sa_temp = SA_TEMPLATE
        self.sq_temp = SQ_TEMPLATE
        self.su_temp = SUM_TEMPLATE
        self.prop_temp_wo = PROP_TEMPLATE_WO
        self.dec_temp = DEC_TEMPLATE
        self.su_temp_prop = SUM_TEMPLATE_PROP

    # 与原先稍作改动，利用了get_action_image_combinations函数获取所有图像
    # 或者视频的列表，然后批次处理。这里的 get_action_image_combinations 函数还没加第二个列表参数，请注意
    def get_image_tensor(self, img_path):
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        img_path_lst = get_action_image_combinations(img_path)
        batch_size = self.batch_size
        img_list = [os.path.join(img_path, path) for path in img_path_lst]
        
        # 按批次返回
        for i in range(0, len(img_list), batch_size):
            batch_img_list = img_list[i:i + batch_size]
            images = [Image.open(img).convert("RGB") for img in batch_img_list]

            # 固定大小调整
            fixed_size = self.image_size
            images = [img.resize(fixed_size) for img in images]

            image_sizes = [image.size for image in images]
            image_tensor = process_images(images, self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.model.device)

            # 使用 yield 返回当前批次的图片张量和大小
            yield batch_img_list, image_tensor, image_sizes
        #return batch_img_list, image_tensor, image_sizes
    
    # 仿照图像，实现了视频的批量处理，这里的 get_action_image_combinations 函数还没加第二个列表参数，请注意
    # 这里不同于图像批量处理的是，同批量视频内相应的提示是不同的，必须分别插入标签和时间信息
    def get_video_tensor(self, vid_path):
        
        args = self.args
        if not os.path.exists(vid_path):
            raise FileNotFoundError(f"Video file not found: {vid_path}")
        
        vid_path_lst = get_action_image_combinations(vid_path)
        batch_size = self.batch_size
        vid_list = [os.path.join(vid_path, path) for path in vid_path_lst]
        # 按批次返回
        for i in range(0, len(vid_list), batch_size):
            batch_vid_list = vid_list[i:i + batch_size]
            labels = [path.split("/")[-2] for path in batch_vid_list]
            videos = [load_video_same(vid, args)[0] for vid in batch_vid_list]
            video_time = [load_video_same(vid, args)[2] for vid in batch_vid_list]
            frame_time = [load_video_same(vid, args)[1] for vid in batch_vid_list]
            proc_vids = [self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda() for video in videos]
            prop_prompts =[self.prop_temp.format(self.task, DEFAULT_IMAGE_TOKEN, label) for label in labels]
            if args.add_time_instruction:
                tmp = []
                for idx, prop in enumerate(prop_prompts):
                    time_instruciton = f"The video lasts for {video_time[idx]:.2f} seconds, and {len(videos[idx][0])} frames are uniformly sampled from it. These frames are located at {frame_time[idx]}."
                    tmp.append(f'{time_instruciton}\n{prop}')
                prop_prompts = tmp
            yield proc_vids, prop_prompts
        #return proc_vids, prop_prompts
            
    # 原来的图像批量处理代码，还用旧的self.question作为提示
    def inference(self, img_path):
        for batch_img_list, image_tensor, image_sizes in self.get_image_tensor(img_path):
            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], self.question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
            input_ids = input_ids.repeat(self.batch_size, 1)  # repeat num_images times at dim-0

            with torch.inference_mode():
                conts = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    modalities=["image"] * self.batch_size,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
                )
            ipdb.set_trace()
            text_outputs = self.tokenizer.batch_decode(conts, skip_special_tokens=True)
            #yield text_outputs
        return text_outputs

    # 将从对话模板产生提示，对提示进行分词处理，输入语言模型产生回答并解码这一系列操作模块化
    # 封装近 proc_qs函数。if isinstance(qs, list) 部分是针对批量视频处理，但是遇到bug卡住了
    # else部分是针对单个视频处理，运行良好，可以放心调用
    def proc_qs(self, qs, video):

        conv = conv_templates[self.conv_template].copy()
        if isinstance(qs, list):
            prompt = []
            for q in qs:
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], None)
                prompt.append(conv.get_prompt())
            
            if self.tokenizer.pad_token_id is None:
                if "qwen" in self.tokenizer.name_or_path.lower():
                    print("Setting pad token to bos token for qwen model.")
                    self.tokenizer.pad_token_id = 151643

            # 在mm_utils.py中新定义了tokenizer_image_token_batch函数以作批量分词
            all_input_ids = tokenizer_image_token_batch(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                   
            all_input_ids = pad_and_stack(all_input_ids, self.tokenizer.pad_token_id)
            attention_masks = all_input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
            with torch.inference_mode():
                # bug 出现在这里，ValueError: You are attempting to perform batched generation with 
                # padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Qwen2. 
                # Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input

                output_ids = self.model.generate(inputs=all_input_ids, images= torch.stack(video, dim=0), attention_mask=attention_masks, modalities=["video"] * self.batch_size, do_sample=False, temperature=0.0, max_new_tokens=4096, top_p=0.1, num_beams=1, use_cache=True)
            text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            ipdb.set_trace()
        # else 分支没问题
        else:            
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            # if self.tokenizer.pad_token_id is None:
            #     if "qwen" in self.tokenizer.name_or_path.lower():
            #         print("Setting pad token to bos token for qwen model.")
            #         self.tokenizer.pad_token_id = 151643
                    
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
            with torch.inference_mode():
                output_ids = self.model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities=["video"], do_sample=False, temperature=0.0, max_new_tokens=4096, top_p=0.1, num_beams=1, use_cache=True)
            # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            # keywords = [stop_str]
            # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return text_outputs
    
    # 尝试批量推理视频，有bug卡住了
    def inference_video_batch(self, video_path=None):

        args = self.args
        if video_path is None:
            video_path = args.video_path
        for proc_vids, prop_prompts in self.get_video_tensor(video_path):
            self.proc_qs(prop_prompts, proc_vids)

    def inference_video_prop(self, video_path=None, dic=None):

        args = self.args
        if video_path is None:
            video_path = args.video_path
        vid_path_lst = get_action_image_combinations(video_path)
        vid_path_lst = [path for path in vid_path_lst if path in dic]
        vid_list = [os.path.join(video_path, path) for path in vid_path_lst if path in dic]
        assert len(vid_path_lst) == len(vid_list)
        store = {}
        for idx, vid in enumerate(tqdm(vid_list, desc="Processing videos")):
            store[vid_path_lst[idx]] = {}
            label = vid.split("/")[-2]
            video,frame_time,video_time = load_video_same(vid, args)
            print(f"num frames: {len(video)}")
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            video = [video]
            ipdb.set_trace()
            qs = self.su_temp_prop.format(self.task, DEFAULT_IMAGE_TOKEN, store[vid_path_lst[idx]]['proposition'])
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                qs = f'{time_instruciton}\n{qs}'
            ipdb.set_trace()
            text_outputs = self.proc_qs(qs, video)
            store[vid_path_lst[idx]]['summary'] = text_outputs
            print('summary', text_outputs)

    
    # 单个地进行视频推理，运行良好，先无标签产生要点，最后对比要点和标签
    # 看看是否需要分解问题
    def inference_video_decom(self, video_path=None, lst=None):

        args = self.args
        if video_path is None:
            video_path = args.video_path
        #vid_path_lst = get_action_image_combinations(video_path, lst)
        vid_path_lst = get_action_image_combinations(video_path)
        vid_path_lst = [path for path in vid_path_lst if path in lst]
        vid_list = [os.path.join(video_path, path) for path in vid_path_lst if path in lst]
        assert len(vid_path_lst) == len(vid_list)
        #vid_list = [os.path.join(video_path, path) for path in vid_path_lst]
        store = {}
        for idx, vid in enumerate(tqdm(vid_list, desc="Processing videos")):
            assert os.path.join(video_path, vid_path_lst[idx]) == vid
            store[vid_path_lst[idx]] = {}
            label = vid.split("/")[-2]
            video,frame_time,video_time = load_video_same(vid, args)
            # viz_sample_frames = np.random.choice([0, 1], p=[0.8, 0.2])
            # if viz_sample_frames:
            #     # save video images
            #     video_out_dir = 'video_sample_images/'
            #     if not os.path.exists(video_out_dir):
            #         os.makedirs(video_out_dir)
            #     frames = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR) for img in video]
            #     # concat all frames to one big figure
            #     samples = np.concatenate(frames, axis=1)
            #     cv2.imwrite('{}/{}_{}.jpg'.format(video_out_dir, vid.rsplit('/',2)[-2], vid.rsplit('/',2)[-1]), samples)

            # print(f"num frames: {len(video)}")
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            video = [video]

            # Run inference on the video and add the output to the list
            qs = self.prop_temp_wo.format(self.task, DEFAULT_IMAGE_TOKEN)
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                qs = f'{time_instruciton}\n{qs}'

            text_outputs = self.proc_qs(qs, video)            
            store[vid_path_lst[idx]]['proposition'] = text_outputs
            print('proposition', text_outputs)

            qs = self.dec_temp.format(self.task, DEFAULT_IMAGE_TOKEN, label, store[vid_path_lst[idx]]['proposition'], store[vid_path_lst[idx]]['proposition'], label)
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                qs = f'{time_instruciton}\n{qs}'

            text_outputs = self.proc_qs(qs, video)            
            store[vid_path_lst[idx]]['decom'] = text_outputs
            print('decom', text_outputs)

            # 需要分解，则标注自我提问
            if text_outputs.startswith("Yes"):
                qs = self.sq_temp.format(self.task, DEFAULT_IMAGE_TOKEN, store[vid_path_lst[idx]]['proposition'])
                if args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                    qs = f'{time_instruciton}\n{qs}'
                text_outputs = self.proc_qs(qs, video)
                store[vid_path_lst[idx]]['questions'] = text_outputs
                print('questions', text_outputs)

            # 不需要分解，直接基于要点（无标签）产生描述和识别结果
            elif text_outputs.startswith("No"):
                qs = self.su_temp_prop.format(self.task, DEFAULT_IMAGE_TOKEN, store[vid_path_lst[idx]]['proposition'])
                if args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                    qs = f'{time_instruciton}\n{qs}'
                text_outputs = self.proc_qs(qs, video)
                store[vid_path_lst[idx]]['summary'] = text_outputs
                print('summary', text_outputs)

        return store
    
    # 单个地进行视频推理，运行良好。先基于标签产生权威的要点，同时此问题组
    # 也是完整推理流程中的自问部分，然后就是自答和最终描述一气呵成
    def inference_video_full(self, video_path=None, dic=None):
        """
        Inference on a video file.
        """
        args = self.args
        if video_path is None:
            video_path = args.video_path
        vid_path_lst = get_action_image_combinations(video_path)
        vid_path_lst = [path for path in vid_path_lst if path in dic]
        vid_list = [os.path.join(video_path, path) for path in vid_path_lst if path in dic]
        assert len(vid_path_lst) == len(vid_list)
        store = {}
        for idx, vid in enumerate(tqdm(vid_list, desc="Processing videos")):
            assert os.path.join(video_path, vid_path_lst[idx]) == vid
            store[vid_path_lst[idx]] = {}
            label = vid.split("/")[-2]
            video,frame_time,video_time = load_video_same(vid, args)
            # viz_sample_frames = np.random.choice([0, 1], p=[0.8, 0.2])
            # if viz_sample_frames:
            #     # save video images
            #     video_out_dir = 'video_sample_images/'
            #     if not os.path.exists(video_out_dir):
            #         os.makedirs(video_out_dir)
            #     frames = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR) for img in video]
            #     # concat all frames to one big figure
            #     samples = np.concatenate(frames, axis=1)
            #     cv2.imwrite('{}/{}_{}.jpg'.format(video_out_dir, vid.rsplit('/',2)[-2], vid.rsplit('/',2)[-1]), samples)
            
            print(f"num frames: {len(video)}")
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            video = [video]

            
            # Run inference on the video and add the output to the list
            qs = self.prop_temp.format(self.task, DEFAULT_IMAGE_TOKEN, label)
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                qs = f'{time_instruciton}\n{qs}'
            # if self.model.config.mm_use_im_start_end:
            #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            # else:
            #     qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            text_outputs = self.proc_qs(qs, video)            
            store[vid_path_lst[idx]]['proposition'] = text_outputs
            print('proposition', text_outputs)

            
            qs = self.sq_temp.format(self.task, DEFAULT_IMAGE_TOKEN, store[vid_path_lst[idx]]['proposition'])
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                qs = f'{time_instruciton}\n{qs}'
            text_outputs = self.proc_qs(qs, video)
            store[vid_path_lst[idx]]['questions'] = text_outputs
            print('questions', text_outputs)

            qs = self.sa_temp.format(DEFAULT_IMAGE_TOKEN, store[vid_path_lst[idx]]['questions'], self.task)
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                qs = f'{time_instruciton}\n{qs}'
            text_outputs = self.proc_qs(qs, video)
            store[vid_path_lst[idx]]['answers'] = text_outputs
            print('answers', text_outputs)

            qs = self.su_temp.format(self.task, DEFAULT_IMAGE_TOKEN, store[vid_path_lst[idx]]['questions'], store[vid_path_lst[idx]]['answers'])
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
                qs = f'{time_instruciton}\n{qs}'
            text_outputs = self.proc_qs(qs, video)
            store[vid_path_lst[idx]]['summary'] = text_outputs
            print('summary', text_outputs)

        return store

# 这是为了单个处理视频提速用的，先将所有动作进行均分，然后并行处理小份
def split_list(lst, n):
    """将列表lst大约均分为n个子列表"""
    chunk_size = math.ceil(len(lst) / n)  # 计算每个子列表的大小
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

if __name__ == "__main__":
    args = parse_args()

    llava = LLaVANeXTEngine(args,
        pretrained="path to OV72B",
    )
    # 图像根路径，下面是 动作/图像名 组合
    root_path = "root frames path"
    # 视频根路径，下面是 动作/视频名 组合
    video_root = "root videos path"

    # 给每份动作视频进行分解标注
    # outputs = llava.inference_video_decom(video_root, splitted[3])
    with open(os.path.join(video_root, 'yes_summary_questions-NovSup.json'), "r") as f:
        data = json.load(f)
    
    outputs = llava.inference_video_full(video_root, data)
    # 保存标注结果
    with open("{}/{}.json".format(video_root, 'ov-vid-full-anno-novsup'), "w") as f:
        json.dump(outputs, f)

