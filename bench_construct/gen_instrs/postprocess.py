import os
world_size = int(os.environ['SLURM_NTASKS']  )
node_id = os.environ['SLURM_NODEID']   
rank = int(os.environ['SLURM_PROCID']   )
local_rank = int(os.environ['SLURM_LOCALID'])
node_list = os.environ['SLURM_NODELIST']
print(f"rank: {rank}, world_size: {world_size}, node_id: {node_id}, local_rank: {local_rank}")
import gzip 
import json
from tqdm import tqdm
import random
import numpy as np
import types
from omegaconf import OmegaConf
import re

from utils import get_ep_goals

global_config = dict(
    DATA_DIR = "/mnt/hwfile/zhangsiqi1/project/vlfm/data",
    SCENE_INFO_PATH = '/mnt/inspurfs/efm_t/huangwensi/vl_ln/vln_llava_data/scene_summary',
    MP3D_SCENE = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/scene_datasets/mp3d",
    VIEW_POINT = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/object_viewpoints_1107",
    REGIONS = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/scene_floors_sim_mp3d_all.json"
)
test_config = dict(
    DS_DIR = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_test/tour_rgb",
    TRAJ_DIR = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_test/shortest_traverse_path",
    REGION_DIST = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/meta_data/scene_regions_dists_test.json",
    
    output_dir = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_test/instructions"
)
suff = ""
# suff = '_rep1'
train_config = dict(
    DS_DIR = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/tour_rgb{suff}",
    TRAJ_DIR = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/shortest_traverse_path{suff}",
    REGION_DIST = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/meta_data/scene_regions_dists_train.json",
    
    output_dir = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/instructions{suff}"
)


def _check_count_cur(ori_instr, reph_instr):
    if 'prior' in ori_instr:
        pattern = r"prior.*?arrival"
    else:
        pattern = r"following.*?entry"
    match_result = re.search(pattern, ori_instr)
    extracted_text = match_result.group()
    words = extracted_text.split(' ')
    key = words[-2]
    if key in reph_instr:
        return True
    else:
        return False
    
def _check_count_other(ori_instr, reph_instr):
    pattern = r"the.*?that"
    match_result = re.search(pattern, ori_instr)
    extracted_text = match_result.group(1)  # second bedroom
    words = extracted_text.split(' ')
    key = words[0]
    if key in reph_instr:
        return True
    else:
        return False
    

def check_instr(ori_instr, reph_instr, task_type):
    if task_type == 'room_temporal_count/cur_room':
        valid = _check_count_cur(ori_instr, reph_instr)
    elif task_type == 'room_temporal_order/multi':
        valid = False
    elif task_type == 'extracted_text':
        valid = _check_count_other(ori_instr, reph_instr)
    else:
        valid = True
        
    if valid == False:
        return ori_instr
    else:
        return reph_instr

def cat_postprocess_test(instr_dir):
    with gzip.open(f"{instr_dir}/all.json.gz", 'r') as f:
        ori_data = json.load(f)
    rephrased_instrs = json.load(open(f"{instr_dir}/rephrased_instructions.json", 'r'))
        
    new_data = {
        'task_to_idx': ori_data['task_to_idx'],
        'episodes': [],
        'category_to_task_category_id': {},
        'category_to_scene_annotation_category_id': {}
    }
    
    cat_eps = []
    for ep in ori_data['episodes']:
        epid = ep['episode_id']
        if epid not in rephrased_instrs:
            new_data['episodes'].append(ep)
        else:
            reph_instr = rephrased_instrs[ep['episode_id']]
            _instr = ep['instruction']
            final_instr = check_instr(_instr['instruction_text'], reph_instr, _instr['task_type'])
            ep['instruction']['instruction_text'] = final_instr
        
            if ep['instruction']['task_type'] == 'object_category':
                cat_eps.append(ep)
            else:
                new_data['episodes'].append(ep)
    
    print(len(cat_eps))
    scans = [x['scene_id'].split('/')[1] for x in cat_eps]
    scans = list(set(scans))
    
    for scan in scans:
        objects_dict = json.load(open(os.path.join(global_config['SCENE_INFO_PATH'], scan, 'object_dict.json'), 'r'))
        inst_viewpoints = json.load(open(os.path.join(global_config['VIEW_POINT'], scan+'.json'), 'r'))
        
        scan_dir = os.path.join(test_config['DS_DIR'], scan)
        heights = [x for x in os.listdir(scan_dir) if '.json' not in x and '_' not in x]

        for hkey in heights:
            ds_rooms = json.load(open(f"{scan_dir}/rooms_{hkey}.json", 'r'))
            ds_detailed_rooms = ds_rooms['detail']
            
            scanh_eps = [x for x in cat_eps if scan in x['scene_id'] and x['hkey']==hkey]
            for ep in tqdm(scanh_eps, desc=f"{scan}|{hkey}"):
                start_idx, end_idx = ep['start_idx'], ep['end_idx']
                detailed_rooms = ds_detailed_rooms[start_idx : end_idx]
                room_set = set([tuple(x) for x in detailed_rooms if x!=[]])

                target_inst = ep['instruction']['instance_id'][0]
                target_cat = ep['object_category']
                cat_instances = []
                for room in list(room_set):
                    rid = room[0].split('_')[-1]
                    cat_room_insts = [
                        k for k,v in objects_dict.items()
                        if f"region{rid}/" in k
                        and v['category'] == target_cat
                        and k != target_inst
                    ]
                    cat_instances += cat_room_insts
                    
                ep['goals'] = get_ep_goals(
                    [target_inst] + cat_room_insts, objects_dict, inst_viewpoints
                )
                new_data['episodes'].append(ep)
                    
    with gzip.open(f"{instr_dir}/final.json.gz", 'w') as f:
        print(len(new_data['episodes']))
        f.write(json.dumps(new_data).encode('utf-8'))
                
def postprocess_train(instr_dir):
    '''
    检查rephrase instr中是否包含需要保留的重要信息
    替换为rephrase instr后存储为final.json.gz
    '''
    ori_eps = []
    for file in ['instr_nocat.json.gz', 'instr_cat.json.gz']:
        with gzip.open(f"{instr_dir}/{file}", 'r') as f:
            ori_data = json.load(f)
            ori_eps += ori_data['episodes']
    rephrased_instrs = json.load(open(f"{instr_dir}/rephrased_instructions.json", 'r'))
        
    new_data = {
        'task_to_idx': ori_data['task_to_idx'],
        'episodes': [],
        'category_to_task_category_id': {},
        'category_to_scene_annotation_category_id': {}
    }
    
    for ep in tqdm(ori_eps):
        epid = ep['episode_id']
        if epid not in rephrased_instrs:
            new_data['episodes'].append(ep)
        else:
            reph_instr = rephrased_instrs[ep['episode_id']]
            _instr = ep['instruction']
            final_instr = check_instr(_instr['instruction_text'], reph_instr, _instr['task_type'])
            ep['instruction']['instruction_text'] = final_instr
            new_data['episodes'].append(ep)
    print('total eps:', len(new_data['episodes']))
    
    with gzip.open(f"{instr_dir}/final.json.gz", 'w') as f:
        f.write(json.dumps(new_data).encode('utf-8'))
    
    
if __name__=='__main__':
    # test_instr_dir = '/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_test/instructions'
    # cat_postprocess_test(test_instr_dir)
    
    train_instr_dir = '/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/instructions_rep3'
    postprocess_train(train_instr_dir)