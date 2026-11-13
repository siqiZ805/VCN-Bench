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
from constants import *
from spatial import *
from temporal import *
from object import *

import habitat_sim
import sys
sys.path.append('..')
from habitat_utils.sim import HabitatSimulator
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

global_config = dict(
    # DATA_DIR = "/mnt/hwfile/zhangsiqi1/project/vlfm/data",
    SCENE_INFO_PATH = '/mnt/data/oss/zhangsiqi/VLLN-Bench/raw_data/mp3d/scene_summary',
    MP3D_SCENE = "/mnt/data/cpfs/b5/navigation_data/internnav_data/data/scene_data/mp3d",
    VIEW_POINT = "/mnt/data/oss/zhangsiqi/object_viewpoints_1107",
    REGIONS = f"/mnt/hazel_zhang_nas_Bzone/project/3D-Mem/myFam/mp3d_human_test/meta_data/scene_floors_sim_mp3d_all.json"
)

test_config = dict(
    DS_DIR = "/mnt/hazel_zhang_nas_Bzone/project/3D-Mem/myFam/mp3d_human_test/tour_rgb",
    TRAJ_DIR = "/mnt/hazel_zhang_nas_Bzone/project/3D-Mem/myFam/mp3d_human_test/shortest_traverse_path",
    REGION_DIST = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/meta_data/scene_regions_dists_test.json",
    
    output_dir = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_test/instructions"
)

# suff = ""
suff = '_rep3'
train_config = dict(
    DS_DIR = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/tour_rgb{suff}",
    TRAJ_DIR = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/shortest_traverse_path{suff}",
    REGION_DIST = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/meta_data/scene_regions_dists_train.json",
    
    output_dir = f"/mnt/hwfile/zhangsiqi1/project/vlfm/data/datasets/myFam/mp3d_human_train/instructions{suff}"
)

def construct_sim(scan):
    global global_config
    sim_settings = OmegaConf.load('../habitat_utils/sim_config.yaml')
    agent_setting = OmegaConf.load('../habitat_utils/agent_config.yaml')
    sim_settings['scene'] = os.path.join(global_config['MP3D_SCENE'], scan, scan+'.glb')
    abstract_sim = HabitatSimulator(sim_settings, agent_setting)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    pathfinder  = sim.pathfinder
    return sim, agent, pathfinder


def main(config, task_type, func, mode, reverse=None):
    results = {
        'episodes': [],
    }
    scans = os.listdir(config.DS_DIR)
    # scans = ['8194nk5LbLH', 'EU6Fwq7SyZv', 'zsNo4HB9uLZ', 'oLBMNvg9in8', 'pLe4wQe7qrG', 'Z6MFQCViBuw']
    scene_region_dists = json.load(open(config.REGION_DIST, 'r'))
    all_regions = json.load(open(config.REGIONS, 'r'))
    
    if task_type in ['obj_instance', 'obj_category']:
        det_step = 50
        window_step = 100
    else:
        det_step = 1
        window_step = 10
    
    for scan in tqdm(scans[rank::world_size]):
        if scan in test_scans:
            continue
        if task_type == 'obj_category':
            if os.path.exists(os.path.join(config.output_dir, task_type+'_'+scan+'.json')):
                continue
            try:
                sim.close()
            except:
                pass
            sim, agent, pathfinder = construct_sim(scan)
            results = { 'episodes': [], }
            
        objects_dict = json.load(open(os.path.join(config.SCENE_INFO_PATH, scan, 'object_dict.json'), 'r'))
        inst_viewpoints = json.load(open(os.path.join(config.VIEW_POINT, scan+'.json'), 'r'))
        inst_instructions = json.load(open(os.path.join(config.SCENE_INFO_PATH, scan, 'object_instruction_with_room.json'), 'r'))
        
        scan_dir = os.path.join(config.DS_DIR, scan)
        heights = [x for x in os.listdir(scan_dir) if '.json' not in x and '_' not in x]
        
        for hkey in heights:
            ori_trajectory = json.load(open(f"{config.TRAJ_DIR}/{scan}_{hkey}.json", 'r'))['trajectory']
            ori_rooms = json.load(open(f"{config.TRAJ_DIR}/{scan}_{hkey}_rooms.json", 'r'))
            ori_detailed_rooms = ori_rooms['detail']
            
            ds_trajectory = json.load(open(f"{scan_dir}/prior_{hkey}.json", 'r'))['trajectory']
            ds_rooms = json.load(open(f"{scan_dir}/rooms_{hkey}.json", 'r'))
            ds_detailed_rooms = ds_rooms['detail']
            ds_abs_rooms = ds_rooms['abstract']
            ds_insts = json.load(open(f"{scan_dir}/instances_{hkey}.json", 'r'))
            
            all_episodes = []
            det_point = 0
            for abs_idx, start_room in enumerate(ds_abs_rooms):
                while det_point < len(ds_detailed_rooms) and ds_detailed_rooms[det_point] != start_room:
                    det_point += det_step
                if det_point >= len(ds_detailed_rooms) - 10:
                    break
                
                max_traj_len = len(ds_detailed_rooms) - det_point
                cand_lens = np.arange(10, max_traj_len, window_step)
                
                for traj_len in cand_lens:
                    end_idx = det_point + traj_len
                    if end_idx >= len(ds_detailed_rooms):
                        continue
                    input_dict = {
                        'ori_trajectory': ori_trajectory,
                        'ori_detailed_rooms': ori_detailed_rooms,
                        'trajectory': ds_trajectory[det_point : end_idx],
                        'detailed_rooms': ds_detailed_rooms[det_point : end_idx],
                        'traj_insts': ds_insts[det_point : end_idx],
                    }
                    input_dict['abs_rooms'] = abstract_traj(input_dict['detailed_rooms'])
                    room_set = set([tuple(x) for x in input_dict['abs_rooms']])
                    if len(list(room_set)) < 2:
                        continue
                    input_dict.update({
                        'room_set': list(room_set),
                        'objects_dict': objects_dict,
                        'inst_viewpoints': inst_viewpoints,
                        'inst_instructions': inst_instructions,
                        'scene_region_dists': scene_region_dists[scan],
                        'all_regions': all_regions[scan],
                        # 'scene_NSWE': scene_NSWE[scan],
                        'scan': scan,
                    })
                    #*-----------------------------
                    if task_type == 'obj_category':
                        episodes = func(input_dict, pathfinder)
                    else:
                        episodes = func(input_dict, mode)
                    
                    for ep in episodes:
                        ep.update({
                            'scene_id': f"mp3d/{scan}/{scan}.glb",
                            'hkey': hkey,
                            'start_idx': det_point,
                            'end_idx': int(end_idx)
                        })
                        all_episodes.append(ep)
                    
            all_tags = set()
            random.shuffle(all_episodes)
            for ep in all_episodes:
                tag = ep['tag']
                if tag in all_tags:
                    continue
                all_tags.add(tag)
                results['episodes'].append(ep)
            
        if task_type == 'obj_category':
            with open(os.path.join(config.output_dir, task_type+'_'+scan+'.json'), 'w') as f:
                print(task_type, scan, len(results['episodes']))
                json.dump(results, f)
                
    if task_type != 'obj_category':
        with open(os.path.join(config.output_dir, task_type+'.json'), 'w') as f:
            print(task_type, len(results['episodes']))
            json.dump(results, f)


def merge_and_validate(config):
    files = [
        x for x in os.listdir(config.output_dir)
        if x.endswith('.json')
    ]
    all_episodes = []
    for file in files:
        data = json.load(open(os.path.join(config.output_dir, file), 'r'))['episodes']
        if 'category' not in file:
            print(file, len(data))
            random.shuffle(data)
            all_episodes += data[:500]
        else:
            all_episodes += data
    
    # task_types = set([x['instruction']['task_type'] for x in all_episodes])
    # task_to_idx = {tt:idx for idx,tt in enumerate(list(task_types))}
    print(task_to_idx)
    task_count = {tt:0 for tt in list(task_to_idx)}
    high_task_types = list(set(
        [x.split('/')[0] for x in list(task_to_idx)]
    ))
        
    new_episodes = []
    scans = os.listdir(config.DS_DIR)
    for scan in scans:
        heights = [x for x in os.listdir(os.path.join(config.DS_DIR, scan))
                   if '.json' not in x and '_' not in x]
        try:
            sim.close()
        except:
            pass
        sim, agent, pathfinder = construct_sim(scan)
        
        for hkey in heights:
            prior = json.load(open(
                os.path.join(config.DS_DIR, scan, f"prior_{hkey}.json"), 'r'
            ))
            scanh_episodes = [x for x in all_episodes 
                             if scan in x['scene_id']
                             and x['hkey'] == hkey]
            if len(scanh_episodes) == 0:
                continue
            
            start_state = prior['trajectory'][0]
            agent_state = habitat_sim.AgentState()
            agent_state.position = start_state['position']
            agent_state.rotation = start_state['rotation']
            agent.set_state(agent_state)
            
            for ep in tqdm(scanh_episodes, desc=f"{scan} {hkey}"):
                tt = ep['instruction']['task_type']
                task_count[tt] += 1
                
                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = ep['start_position']
                path.requested_ends = [x['agent_state']['position'] 
                                        for goal in ep['goals']
                                        for x in goal['view_points']
                                    ]
                pathfinder.find_path(path)
                dist = path.geodesic_distance
                if dist < 3 or dist == np.inf:
                    continue
                ep['dist_to_goal'] = dist
                ep['episode_id'] = f"{task_to_idx[tt]}_{task_count[tt]}"
                new_episodes.append(ep)
                
    select_episodes = []
    for high_tt in high_task_types:
        if high_tt == 'room_temporal_count':
            cur_room_eps = [x for x in new_episodes if x['instruction']['task_type']=='room_temporal_count/cur_room']
            other_room_eps = [x for x in new_episodes if x['instruction']['task_type']=='room_temporal_count/other_room']
            random.shuffle(other_room_eps)
            select_episodes += cur_room_eps
            select_episodes += other_room_eps[: 250-len(cur_room_eps)]
        else:
            tt_eps = [x for x in new_episodes if high_tt in x['instruction']['task_type']]
            random.shuffle(tt_eps)
            select_episodes += tt_eps[:250]
    
    results = {
        'task_to_idx': task_to_idx,
        'episodes': select_episodes,
        'category_to_task_category_id': {},
        'category_to_scene_annotation_category_id': {}
    }
    epids = list(set([x['episode_id'] for x in select_episodes]))
    print(len(select_episodes), len(epids))
    assert len(select_episodes) == len(epids)
    final_task_count = Counter([x['instruction']['task_type'] for x in select_episodes])
    print(final_task_count)
    with gzip.open(f"{config.output_dir}/all.json.gz", 'w') as f:
        f.write(json.dumps(results).encode('utf-8'))
    

def merge_and_validate_train(config, do_cat=False):
    files = [
        x for x in os.listdir(config.output_dir)
        if x.endswith('.json')
        and 'rephrase' not in x
    ]
    all_episodes = []
    # cat_episodes = []
    for file in files:
        if do_cat == False:
            if 'category' not in file:
                data = json.load(open(os.path.join(config.output_dir, file), 'r'))['episodes']
                print(file, len(data))
                all_episodes += data[:10000]
        else:
            if 'category' in file:
                data = json.load(open(os.path.join(config.output_dir, file), 'r'))['episodes']
                print(file, len(data))
                all_episodes += data
    if do_cat:
        print('obj_category:', len(all_episodes))
    
    print(task_to_idx)
    task_count = {tt:0 for tt in list(task_to_idx)}
    high_task_types = list(set(
        [x.split('/')[0] for x in list(task_to_idx)]
    ))
    
    new_episodes = []
    scans = os.listdir(config.DS_DIR)
    for scan in scans:
        heights = [x for x in os.listdir(os.path.join(config.DS_DIR, scan))
                   if '.json' not in x and '_' not in x]
        try:
            sim.close()
        except:
            pass
        sim, agent, pathfinder = construct_sim(scan)
        
        for hkey in heights:
            prior = json.load(open(
                os.path.join(config.DS_DIR, scan, f"prior_{hkey}.json"), 'r'
            ))
            scanh_episodes = [x for x in all_episodes 
                             if scan in x['scene_id']
                             and x['hkey'] == hkey]
            if len(scanh_episodes) == 0:
                continue
            
            start_state = prior['trajectory'][0]
            agent_state = habitat_sim.AgentState()
            agent_state.position = start_state['position']
            agent_state.rotation = start_state['rotation']
            agent.set_state(agent_state)
            
            for ep in tqdm(scanh_episodes, desc=f"{scan} {hkey}"):
                tt = ep['instruction']['task_type']
                task_count[tt] += 1
                
                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = ep['start_position']
                path.requested_ends = [x['agent_state']['position'] 
                                        for goal in ep['goals']
                                        for x in goal['view_points']
                                    ]
                pathfinder.find_path(path)
                dist = path.geodesic_distance
                if dist < 3 or dist == np.inf:
                    continue
                ep['dist_to_goal'] = dist
                ep['episode_id'] = f"{task_to_idx[tt]}_{task_count[tt]}"
                new_episodes.append(ep)
                
    results = {
        'task_to_idx': task_to_idx,
        'episodes': new_episodes,
        'category_to_task_category_id': {},
        'category_to_scene_annotation_category_id': {}
    }
    epids = list(set([x['episode_id'] for x in new_episodes]))
    print(len(new_episodes), len(epids))
    assert len(new_episodes) == len(epids)
    final_task_count = Counter([x['instruction']['task_type'] for x in new_episodes])
    print(final_task_count)
    
    if do_cat:
        output_path = f"{config.output_dir}/instr_cat.json.gz"
    else:
        output_path = f"{config.output_dir}/instr_nocat.json.gz"
    with gzip.open(output_path, 'w') as f:
        f.write(json.dumps(results).encode('utf-8'))
    


    

if __name__=='__main__':
    mode = 'train' #!
    config = global_config
    if mode == 'test':
        config.update(test_config)
    else:
        config.update(train_config)
    config = types.SimpleNamespace(**config)
    if rank == 0 and not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    
    # main(config, 'spatial_distance', spatial_distance, mode, reverse=None)
    # main(config, 'spatial_orientation', spatial_orientation_rotation, mode, reverse=None)
    
    # main(config, 'temporal_count', temporal_count, mode, reverse=None)
    # main(config, 'temporal_order_multi', temporal_order_multi, mode, reverse=None)
    # main(config, 'temporal_order_sing', temporal_order_sing, mode, reverse=None)
    
    # main(config, 'obj_instance', obj_instance, mode)
    # main(config, 'obj_category', obj_category, mode)
    # srun -p efm_t --unbuffered --label --gres=gpu:1 -N 1 --ntasks=24 python -u run.py
    
    # merge_and_validate(config)
    merge_and_validate_train(config, do_cat=True)