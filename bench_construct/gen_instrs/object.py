from collections import Counter
import numpy as np
import random
from tqdm import tqdm

from constants import *
from utils import *

def obj_instance(input_dict, mode='test'):
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    scene_region_dists, all_regions = input_dict['scene_region_dists'], input_dict['all_regions']
    traj_insts = input_dict['traj_insts']
    
    episodes = []
    
    vis_insts = sum([list(x) for x in traj_insts], [])
    vis_insts_winstr = [x for x in vis_insts if x in inst_instructions]
    if vis_insts_winstr == []:
        return []
    
    for inst in vis_insts_winstr:
        cat = objects_dict[inst]['category']
        if cat in skip_objects:
            continue
        instruction = inst_instructions[inst]
        print(instruction)
        inst_rid = inst.split('/')[0].split('region')[-1]
        # target_room  = [x for x in room_set if f"_{inst_rid}" in x[1]]
        # if len(target_room) != 1:
        #     import pdb; pdb.set_trace()
        # target_room = target_room[0]
        
        cand_init_pos_room = [
            (traj_item['position'], list(room))
            for traj_item, room
            in zip(ori_trajectory, ori_detailed_rooms)
            if room != [] and f"_{inst_rid}" not in room[1]
            and list(room) in detailed_rooms
        ]
        if cand_init_pos_room == []:
            continue
        init_pos, init_room = random.choice(cand_init_pos_room)
        
        frame_idxs = [idx for idx, inst_dict in enumerate(traj_insts)
                if inst_dict != {} and inst in list(inst_dict)]
        if frame_idxs == []:
            continue
    
        episode = {
            'start_room': init_room,
            'start_position': init_pos,
            'start_rotation': random_yaw_quaternion_list(),
            'instruction': {
                'task_type': 'object_instance',
                'instruction_text': instruction,
                'instance_id': [inst]
            },
            'object_category': cat,
            'frame_idxs': frame_idxs,
            # 'target_room': list(target_room),
            'goals': get_ep_goals([inst], objects_dict, inst_viewpoints),
            'tag': (init_room[0], inst, input_dict['scan'])
        }
        episodes.append(episode)
    return episodes


def obj_category(input_dict, pathfinder):
    import habitat_sim
    
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    scene_region_dists, all_regions = input_dict['scene_region_dists'], input_dict['all_regions']
    traj_insts = input_dict['traj_insts']
    
    episodes = []
    
    vis_insts = sum([list(x) for x in traj_insts], [])
    if vis_insts == []:
        return []
    vis_insts = list(set(vis_insts))
    vis_insts_cat = [objects_dict[x]['category'] for x in vis_insts]
    
    all_instances = {}
    for room in room_set:
        rid = room[0].split('_')[-1]
        room_insts = {
            k:v for k,v in objects_dict.items()
            if f"region{rid}/" in k
            and v['category'] not in skip_objects
        }
        all_instances.update(room_insts)
    if all_instances == {}:
        return []
    
    all_cats = list(set([v['category'] for k,v in all_instances.items()]))
    for target_cat in tqdm(all_cats):
        cat_vis_insts = [inst
            for inst, cat in zip(vis_insts, vis_insts_cat)
            if cat == target_cat
        ]
        if cat_vis_insts == []:
            continue
        
        #* init room 中没有这个类别的物体
        cat_insts = [k for k,v in all_instances.items() if v['category']==target_cat]
        cat_insts_rid = [x.split('/')[0].split('region')[-1] for x in cat_insts]
        cand_init_rooms = [list(x) for x in room_set if x[0].split('_')[-1] not in cat_insts_rid]
        if cand_init_rooms == []:
            continue
        
        for init_room in cand_init_rooms:
            cand_init_pos = [
                traj_item['position']
                for traj_item, room in zip(ori_trajectory, ori_detailed_rooms)
                if room != [] and room == init_room
            ]
            if cand_init_pos == []:
                continue
            
            #* 看到的那个instance 得是 这个cat所有物体中 离的最近的
            for target_instance in cat_vis_insts:
                frame_idxs = [idx for idx, inst_dict in enumerate(traj_insts)
                    if inst_dict != {} and target_instance in list(inst_dict)]
                if frame_idxs == []:
                    continue
                # print(target_instance)
                # import pdb; pdb.set_trace()
                for init_pos in cand_init_pos:
                    path = habitat_sim.MultiGoalShortestPath()
                    path.requested_start = init_pos
                    path.requested_ends = inst_viewpoints[target_instance]['view_points']
                    pathfinder.find_path(path)
                    target_dist = path.geodesic_distance
                    if target_dist <= 3:
                        continue
                    if target_dist == np.inf:
                        continue
                    
                    other_viewpoints = []
                    for inst in cat_insts:
                        if inst == target_instance:
                            continue
                        other_viewpoints += inst_viewpoints[inst]['view_points']
                    if other_viewpoints == []:
                        other_dist = target_dist + 1000
                    else:
                        path.requested_ends = other_viewpoints
                        pathfinder.find_path(path)
                        other_dist = path.geodesic_distance
                    # print(target_dist, other_dist)
                    
                    if target_dist == np.inf or other_dist == np.inf:
                        # import pdb; pdb.set_trace()
                        continue
                    
                    if target_dist < other_dist - 3:
                        print(target_dist, other_dist)
                        # print('----------------------')
                        episode = {
                            'start_room': init_room,
                            'start_position': init_pos,
                            'start_rotation': random_yaw_quaternion_list(),
                            'instruction': {
                                'task_type': 'object_category',
                                'instruction_text': target_cat,
                                'instance_id': [target_instance],
                            },
                            'object_category': target_cat,
                            'frame_idxs': frame_idxs,
                            'goals': get_ep_goals([target_instance], objects_dict, inst_viewpoints),
                            'tag': (init_room[0], target_instance, input_dict['scan'])
                        }
                        episodes.append(episode)
                        break
    return episodes
    