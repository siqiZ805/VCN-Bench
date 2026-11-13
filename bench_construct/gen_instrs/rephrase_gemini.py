from openai import OpenAI
KEY = "sk-eizRCcZ5337r7AFoeHndmXuCWTaqHRwe49pDHGjfC3fcmCBT"
import re
import json
import gzip
import copy
import numpy as np
import base64
import random
import argparse
import os
from tqdm import tqdm
world_size = int(os.environ['SLURM_NTASKS']  )
node_id = os.environ['SLURM_NODEID']   
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
node_list = os.environ['SLURM_NODELIST']
print(f"rank: {rank}, world_size: {world_size}, node_id: {node_id}, local_rank: {local_rank}")

client = OpenAI(
    # base_url = "https://api.openai.com/v1",
    base_url="https://api.boyuerichdata.opensphereai.com/v1",
    # base_url="http://34.13.73.248:3888/v1", # 谷歌负载均衡网络，全球节点，适合国外访问
    # base_url="https://api.boyuerichdata.opensphereai.com/v1", # 直连香港https，数据加密
    api_key=KEY
)

PROMPT = """Optimize the navigation instruction to enhance naturalness and task-appropriateness.
The instruction is generated based on a tour video. Do not confuse the video with the agent's own navigation history.
Rather than uniformly using “locate”, the verb should be semantically aligned with the target object's function or intended interaction—for example, “open the cabinet”, “turn on the TV”.
{retain}

Instruction: {ori_instr}
Rephrased Instruction:"""

def prompt_count_cur(instr):
    assert 'prior' in instr or 'following' in instr
    # 正则表达式模式：匹配从 immediately 开始，到 video 结束的所有内容
    # .*? ：非贪婪匹配任意字符（精准截取到第一个 video 为止）
    pattern = r"immediately.*?video"
    # 执行匹配（search：查找第一个匹配的结果）
    match_result = re.search(pattern, instr)
    extracted_text = match_result.group()

    prompt = copy.deepcopy(PROMPT)
    prompt = prompt.format(
        retain = f'Retain "{extracted_text}" information.',
        ori_instr = instr
    )
    return prompt

def prompt_order_multi(instr):
    pattern = r":\s*(.*?)\s*Locate"
    match_result = re.search(pattern, instr)
    cand_rooms = match_result.group(1)
    
    pattern = r"visited.*?video"
    match_result = re.search(pattern, instr)
    extracted_text = match_result.group()
    
    prompt = copy.deepcopy(PROMPT)
    prompt = prompt.format(
        retain = f'Retain "{extracted_text}" and "{cand_rooms}" information.',
        ori_instr = instr
    )
    return prompt

def prompt_order_sing(instr):
    pattern = r"the.*?video"
    match_result = re.search(pattern, instr)
    extracted_text = match_result.group()
    
    prompt = copy.deepcopy(PROMPT)
    prompt = prompt.format(
        retain = f'Retain "{extracted_text}" information.',
        ori_instr = instr
    )
    return prompt

def prompt_distance(instr):
    prompt = copy.deepcopy(PROMPT)
    prompt = prompt.format(
        retain = f'Retain "among the rooms visited in the video" information.',
        ori_instr = instr
    )
    return prompt


def main(instr_path, output_dir):
    with gzip.open(instr_path, 'r') as f:
        data = json.load(f)['episodes']
        
    for ep in tqdm(data[rank::world_size]):
        epid = ep['episode_id']
        output_path = os.path.join(output_dir, f"{epid}.txt")
        if os.path.exists(output_path):
            continue
            
        task_type = ep['instruction']['task_type']
        ori_instr = ep['instruction']['instruction_text']
        if task_type == 'room_temporal_count/cur_room':
            prompt = prompt_count_cur(ori_instr)
        elif task_type == 'room_temporal_order/multi':
            prompt = prompt_order_multi(ori_instr)
        elif task_type == 'room_temporal_order/single':
            prompt = prompt_order_sing(ori_instr)
        elif task_type == 'room_spatial_distance':
            prompt = prompt_distance(ori_instr)
        else:
            prompt = copy.deepcopy(PROMPT)
            prompt = prompt.format(
                retain = 'Retain "in the video" information.',
                ori_instr = ori_instr
            )

        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }]
        response = client.chat.completions.create(
            model="gemini-3.1-flash-lite-preview",
            messages=message,
            temperature=0.7,
            timeout=200  # 设置超时
        )
        # print(response)
        output_text = response.choices[0].message.content
        print(output_text)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)  # 写入字符串
            
def parse_output(instr):
    out = instr.split('Rephrased Instruction:')[-1]
    out = out.replace('*', '')
    out = out.replace('"', '')
    out = out.replace("'", '')
    out = out.strip()
    if out[-1] != '.':
        out = out.split('(')[0].strip()
    if out[-1] != '.' or '\n' in out:
        return None
    return out

def merge_files(instr_dir, output_dir):
    files = os.listdir(output_dir)
    results = {}
    for file in files:
        epid = file.split('.txt')[0]
        with open(os.path.join(output_dir, file), 'r') as f:
            data = f.read()
        parse_instr = parse_output(data)
        if parse_instr is None:
            continue
        print(parse_instr)
        results[epid] = parse_instr
    with open(os.path.join(instr_dir, 'rephrased_instructions.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__=='__main__':
    #**** test *****
    # instr_dir = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_test/instructions/"
    # instr_path = instr_dir + "all.json.gz"
    #**** train *****
    # for rep_idx in range(5):
    for rep_idx in [3]:
        suff = ''
        if rep_idx != 0:
            suff = f"_rep{rep_idx}"
        instr_dir = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/instructions{suff}/"
        instr_path = instr_dir + 'instr_cat.json.gz'
    #*******************************
    
        output_dir = instr_dir + "rephrase_instrs"
        if rank == 0 and not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        main(instr_path, output_dir)
        # merge_files(instr_dir, output_dir)
