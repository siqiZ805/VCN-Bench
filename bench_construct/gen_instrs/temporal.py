from collections import Counter
import numpy as np
import random
import itertools

from constants import *
from utils import *


def temporal_count(input_dict, mode):
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    
    episodes = []
    
    #* 去第x次到当前房间之前/后去到的房间
    for idx, init_room in enumerate(abs_rooms):
        #* init_room 要被多次访问
        if abs_rooms.count(init_room) < 2:
            continue
        cand_init_positions = [
            traj_item['position']
            for traj_item, room in zip(ori_trajectory, ori_detailed_rooms)
            if room != [] and room[0] == init_room[0]
        ]
        if cand_init_positions == []:
            continue
        visited_room_ids = [x[0] for x in abs_rooms[:idx]]
        visited_count = visited_room_ids.count(init_room[0]) # 已经访问过的次数
        if visited_count > 5:
            continue
        
        for task in ['before', 'after']:
            if task == 'before':
                if idx == 0:
                    continue
                target_room = abs_rooms[idx-1]
            else:
                if idx == len(abs_rooms)-1:
                    continue
                target_room = abs_rooms[idx+1]
            if init_room[1] in skip_rooms or target_room[1] in skip_rooms:
                continue
            
            region_idx = target_room[0].split('_')[-1]
            target_insts, target_cat = select_obj_gen_instr(region_idx, objects_dict, inst_instructions)
            if target_cat is None:
                continue
            
            if task == 'before':
                instruction = f"Go to the room visited immediately prior to the {idx_to_cap[visited_count+1]} arrival at the current room in the video. Locate the {target_cat}."
            else:
                instruction = f"Go to the room visited immediately following the {idx_to_cap[visited_count+1]} entry into the current room in the video. Locate the {target_cat}."
            print(task, instruction)
            
            frame_idxs = [idx for idx, room in enumerate(detailed_rooms)
                    if room!=[] and room[0]==target_room[0]]
            if frame_idxs == []:
                continue
            
            episode = {
                'start_room': init_room,
                'start_position': random.choice(cand_init_positions),
                'start_rotation': random_yaw_quaternion_list(),
                'instruction': {
                    'task_type': 'room_temporal_count/cur_room',
                    'instruction_text': instruction,
                    'instance_id': target_insts
                },
                'object_category': target_cat,
                'frame_idxs': frame_idxs,
                'target_room': target_room,
                'goals': get_ep_goals(target_insts, objects_dict, inst_viewpoints),
                'tag': (init_room[0], target_room[0], idx_to_cap[visited_count+1], input_dict['scan'], task)
            }
            episodes.append(episode)
            
    #* 去视频中访问的第x个xx房间
    room_set_name_count = Counter([x[1] for x in room_set])
    for name, count in room_set_name_count.items():
        if count == 1:
            continue
        if name in skip_rooms:
            continue
        #* 这些房间要有明确的访问顺序
        cand_rooms = []
        for room in abs_rooms:
            if room[1] != name:
                continue
            if cand_rooms == []:
                cand_rooms.append(room)
            else:
                if room[0] != cand_rooms[-1][0]:
                    cand_rooms.append(room)
        cand_room_set = list(set([tuple(x) for x in cand_rooms]))
        if len(cand_room_set) < len(cand_rooms):
            continue
        
        for idx, target_room in enumerate(cand_rooms):
            target_room_name = standard_room_name(target_room[1])
            region_idx = target_room[0].split('_')[-1]
            target_insts, target_cat = select_obj_gen_instr(region_idx, objects_dict, inst_instructions)
            if target_cat is None:
                continue
            try:
                appear_time = idx_to_cap[idx+1]
            except:
                continue
            instruction = f"Proceed to the {appear_time} {target_room_name} that was visited in the video. Locate the {target_cat}."
            print(instruction)
        
            cand_init_pos_room = [
                (traj_item['position'], list(room))
                for traj_item, room
                in zip(ori_trajectory, ori_detailed_rooms)
                if room != [] and room[1] != target_room[1]
                and list(room) in detailed_rooms     #? init_room in video
            ]
            if cand_init_pos_room == []:
                continue
            init_pos_room = random.choice(cand_init_pos_room)
            
            frame_idxs = [idx for idx, room in enumerate(detailed_rooms)
                    if room!=[] and room[0]==target_room[0]]
            if frame_idxs == []:
                continue
    
            episode = {
                'start_room': init_pos_room[1],
                'start_position': init_pos_room[0],
                'start_rotation': random_yaw_quaternion_list(),
                'instruction': {
                    'task_type': 'room_temporal_count/other_room',
                    'instruction_text': instruction,
                    'instance_id': target_insts,
                },
                'object_category': target_cat,
                'frame_idxs': frame_idxs,
                'target_room': list(target_room),
                'goals': get_ep_goals(target_insts, objects_dict, inst_viewpoints),
                'tag': (init_pos_room[1][0], target_room[0], appear_time, input_dict['scan'])
            }
            episodes.append(episode)
        
    return episodes


def temporal_order_multi(input_dict, mode):
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    
    episodes = []
    
    room_set_names = [x[1] for x in room_set]
    room_name_count = Counter(room_set_names)
    for idx, init_room in enumerate(abs_rooms):
        #* init_room 只能访问一次
        if abs_rooms.count(init_room) > 1:
            continue
        cand_init_positions = [
            traj_item['position']
            for traj_item, room in
            zip(ori_trajectory, ori_detailed_rooms)
            if room != [] and room[0] == init_room[0]
        ]
        if cand_init_positions == []:
            continue
    
        before_rooms = abs_rooms[: idx]
        before_room_names = [x[1] for x in before_rooms]
        after_rooms = abs_rooms[idx+1 :]
        after_room_names = [x[1] for x in after_rooms]
        
        before_only_names = [x for x in before_room_names
                             if x not in after_room_names
                             and x not in skip_rooms
                             and x != init_room[1]]
        after_only_names = [x for x in after_room_names
                            if x not in before_room_names
                            and x not in skip_rooms
                            and x != init_room[1]]
        
        before_only_rooms = list(set([
            tuple(x) for x in before_rooms if x[1] in before_only_names
        ]))
        after_only_rooms = list(set([
            tuple(x) for x in after_rooms if x[1] in after_only_names
        ]))
        if before_only_rooms == [] or after_only_rooms == []:
            continue
        
        for task in ['before', 'after']:
            '''
            target room 在视频中唯一，可以在所在区间多次出现
            anchor room 可以不唯一，但是不同anchor room的name不可以重复
            '''
            if task == 'before':
                cand_target_rooms = [list(x)
                    for x in before_only_rooms
                    if room_name_count[x[1]] == 1
                ]
                name_set = list(set(after_only_names))
            else:
                cand_target_rooms = [list(x)
                    for x in after_only_rooms
                    if room_name_count[x[1]] == 1
                ]
                name_set = list(set(before_only_names))
                    
            for target_room in cand_target_rooms:
                target_room_id = target_room[0]
                region_idx = target_room_id.split('_')[-1]
                target_room_name = standard_room_name(target_room[1])
                frame_idxs = [idx for idx, room in enumerate(detailed_rooms)
                        if room != [] and room[0] == target_room_id]
                if frame_idxs == []:
                    continue
                
                if mode == 'test':
                    max_count = min(len(name_set), 4)
                else:
                    max_count = len(name_set)
                for count in range(2, max_count):
                    all_combines = itertools.combinations(name_set, count)
                    for combine in all_combines:
                        if target_room[1] in combine:
                            import pdb; pdb.set_trace()
                        anchor_room_names = [standard_room_name(x) for x in combine]
                        room_names = anchor_room_names + [target_room_name]
                        random.shuffle(room_names)
                        
                        target_insts, target_cat = select_obj_gen_instr(region_idx, objects_dict)
                        if target_cat is None:
                            continue
                        instruction = f"Go to the room that was visited {task} arriving at the current region in the video: {', '.join(room_names)}. Locate the {target_cat}."
                        print(instruction)
                        
                        episode = {
                            'start_room': init_room,
                            'start_position': random.choice(cand_init_positions),
                            'start_rotation': random_yaw_quaternion_list(),
                            'instruction': {
                                'task_type': 'room_temporal_order/multi',
                                'instruction_text': instruction,
                                'instance_id': target_insts
                            },
                            'object_category': target_cat,
                            'frame_idxs': frame_idxs,
                            'target_room': target_room,
                            'goals': get_ep_goals(target_insts, objects_dict, inst_viewpoints),
                            'tag': (init_room[0], tuple(sorted(combine+(target_room[1],))), input_dict['scan'])
                        }
                        episodes.append(episode)
    return episodes


def temporal_order_sing(input_dict, mode):
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    traj_insts = input_dict['traj_insts']
    
    episodes = []
    
    room_set_names = [x[1] for x in room_set]
    for idx, init_room in enumerate(abs_rooms):
        init_room = list(init_room)
        #* 1. start position
        #* 为明确定位，init_room不能被多次访问
        if abs_rooms.count(init_room) > 1:
            continue
        cand_init_positions = [
            traj_item['position']
            for traj_item, room in zip(ori_trajectory, ori_detailed_rooms)
            if room != [] and room[0] == init_room[0]
        ]
        if cand_init_positions == []:
            continue
        
        for _idx in range(2):
            if _idx == 0:  # before time node
                cand_rooms = abs_rooms[ : idx]
            else:
                cand_rooms = abs_rooms[idx+1 : ]
            if cand_rooms == []:
                continue
            cand_room_ids = [x[0].split('_')[-1] for x in cand_rooms]
            
            #* 2 房间不唯一，但是在before或after rooms中是唯一的
            cand_room_names = [x[1] for x in cand_rooms]
            for target_room in cand_rooms:
                target_room_id, target_room_name = target_room
                if target_room_name in skip_rooms:
                    continue
                st_room_name = standard_room_name(target_room_name)
                region_idx = target_room_id.split('_')[-1]
                
                room_flag = room_set_names.count(target_room_name) > 1 \
                    and cand_room_names.count(target_room_name) == 1
                if not room_flag:
                    continue
                target_insts, target_cat = select_obj_gen_instr(region_idx, objects_dict, inst_instructions)
                if target_cat is None:
                    continue
                    
                frame_idxs = [idx for idx, room in enumerate(detailed_rooms) 
                              if room!=[] and room[0]==target_room_id]
                if frame_idxs == []:
                    continue
                
                if _idx == 0:
                    instruction = f"Go to the {st_room_name} that was visited before reaching current room in the video. Locate the {target_cat}."
                else:
                    instruction = f"Go to the {st_room_name} that was visited after the current room in the video. Locate the {target_cat}."
                print(instruction)
                
                episode = {
                    'start_room': init_room,
                    'start_position': random.choice(cand_init_positions),
                    'start_rotation': random_yaw_quaternion_list(),
                    'instruction': {
                        'task_type': 'room_temporal_order/single',
                        'instruction_text': instruction,
                        'instance_id': target_insts
                    },
                    'object_category': target_cat,
                    'frame_idxs': frame_idxs,
                    'target_room': target_room,
                    'goals': get_ep_goals(target_insts, objects_dict, inst_viewpoints),
                    'tag': (init_room[0], target_room[0], _idx, input_dict['scan'])
                }
                episodes.append(episode)
    return episodes