from collections import Counter
import numpy as np
import random

from constants import *
from utils import *


def spatial_distance(input_dict, mode='test'):
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    scene_region_dists, all_regions = input_dict['scene_region_dists'], input_dict['all_regions']
    
    episodes = []
    
    room_set_ids = [x[0] for x in room_set]
    room_set_names = [x[1] for x in room_set]
    room_name_count = Counter(room_set_names)
    for tgt_room_name, count in room_name_count.items():
        if count < 2:
            continue
        if tgt_room_name in skip_rooms:
            continue
        st_tgt_name = standard_room_name(tgt_room_name)
        
        cand_target_rooms = [list(x) for x in room_set if x[1]==tgt_room_name]
        cand_target_room_ids = [x[0] for x in cand_target_rooms]
        
        floor = cand_target_room_ids[0].split('_')[0]
        try:
            room_geo_dists = scene_region_dists[floor]
        except:
            continue
        floor_regions = {x['region_id']:x for x in all_regions[floor]['regions']
                         if x['region_id'] in room_set_ids}
        
        for init_room_id, dist_rooms in room_geo_dists.items():
            #* init room 和 target room 都须在视频中访问过
            if init_room_id not in room_set_ids:
                continue
            init_room = [x for x in room_set if x[0]==init_room_id]
            init_room = list(init_room[0])
            
            #* init position 尽量接近init room的中心点
            cand_init_positions = [
                traj_item['position']
                for traj_item, room in
                zip(ori_trajectory, ori_detailed_rooms)
                if room != [] and room[0] == init_room_id
            ]
            if cand_init_positions == []:
                continue
            init_center = floor_regions[init_room_id]['region_center']
            init_position, min_dist = None, np.inf
            for position in cand_init_positions:
                dist = abs(position[0] - init_center[0]) + abs(position[2] - init_center[2])
                if dist < min_dist:
                    min_dist = dist
                    init_position = position
            if init_position is None:
                continue
            
            init_to_tgt_dist = {}
            for rid, dist in dist_rooms.items():
                if rid not in room_set_ids:
                    continue
                if rid in cand_target_room_ids:
                    init_to_tgt_dist[rid] = dist
            if len(init_to_tgt_dist) < 2:
                continue
            dist_values = list(init_to_tgt_dist.values())
            
            for task in ['nearest', 'farthest']:
                if task == 'nearest':
                    #* init_room 与一个anchor room相近，与其他anchor rooms 距离要至少多3m
                    min_dist = min(init_to_tgt_dist.values())
                    not_distinctive = [abs(x-min_dist) <= 5 
                                       for x in init_to_tgt_dist.values()]
                    if sum(not_distinctive) != 1:
                        continue
                    # min_idx = np.argmin(init_to_tgt_dist.values())  #! 顺序会变
                    # target_room_id = list(init_to_tgt_dist)[min_idx]
                    target_room_id = [rid for rid,d in init_to_tgt_dist.items() if d==min_dist][0]
                else:
                    max_dist = max(init_to_tgt_dist.values())
                    not_distinctive = [abs(x-max_dist) <= 3 
                                       for x in init_to_tgt_dist.values()]
                    if sum(not_distinctive) != 1:
                        continue
                    # max_idx = np.argmax(init_to_tgt_dist.values())
                    # target_room_id = list(init_to_tgt_dist)[max_idx]
                    target_room_id = [rid for rid,d in init_to_tgt_dist.items() if d==max_dist][0]
            
                target_room = [x for x in room_set if x[0]==target_room_id]
                target_room = list(target_room[0])
                region_idx = target_room_id.split('_')[-1]
                target_instances, target_cat = select_obj_gen_instr(region_idx, objects_dict)
                if target_cat is None:
                    continue
                instruction = f"Proceed to the {st_tgt_name} that is {task} from the current position among the rooms visited in the video. Locate the {target_cat}."
                print(instruction)
                
                frame_idxs = [idx for idx, room in enumerate(detailed_rooms)
                              if room!=[] and room[0]==target_room_id]
                if frame_idxs == []:
                    continue
                
                episode = {
                    'start_room': init_room, 
                    'start_position': init_position,
                    'start_rotation': random_yaw_quaternion_list(),
                    'instruction': {
                        'task_type': 'room_spatial_distance',
                        'instruction_text': instruction,
                        'instance_id': target_instances
                    },
                    'object_category': target_cat,
                    'frame_idxs': frame_idxs,
                    'target_room': target_room,
                    'goals': get_ep_goals(target_instances, objects_dict, inst_viewpoints),
                    'tag': (init_room[0], target_room[0], task, input_dict['scan'])
                }
                episodes.append(episode)
    return episodes



def get_nswe(floor_regions, traj_room_set):
    '''
    +z为南，+x为东
    '''
    floor_regions = {x['region_id']:x for x in floor_regions}
    traj_room_info = []
    for room in traj_room_set:
        rinfo = floor_regions[room[0]]
        center = rinfo['region_center']
        sizes = rinfo['region_sizes']
        bbox = [
            center[0] - sizes[0]/2,
            center[0] + sizes[0]/2,
            center[2] - sizes[2]/2,
            center[2] + sizes[2]/2,
        ]  # [x_min, x_max, z_min, z_max]
        traj_room_info.append({
            'region_id': room[0],
            'region_name': rinfo['region_name'],
            'center': center,
            'sizes': sizes,
            'bbox': bbox
        })
    
    all_z_max = [x['bbox'][-1] for x in traj_room_info]
    most_max_z = max(all_z_max)
    south_rooms = [x for x in traj_room_info
                   if most_max_z-5 < x['bbox'][-1] <= most_max_z]
    valid_south_rooms = []  # region_id
    for room in south_rooms:
        more_south_rooms = [
            x for x in traj_room_info 
            if x['center'][2] > room['center'][2]
            and x['bbox'][0] < max(room['bbox'][1]-1, room['center'][0])
            and x['bbox'][1] > min(room['bbox'][0]+1, room['center'][0])
        ]
        if more_south_rooms == []:
            valid_south_rooms.append([room['region_id'], room['region_name']])
    
    all_z_min = [x['bbox'][-2] for x in traj_room_info]
    most_min_z = min(all_z_min)
    north_rooms = [x for x in traj_room_info
                   if most_min_z <= x['bbox'][-2] < most_min_z+5]
    valid_north_rooms = []
    for room in north_rooms:
        more_north_rooms = [
            x for x in traj_room_info
            if x['center'][2] < room['center'][2]
            and x['bbox'][0] < max(room['bbox'][1]-1, room['center'][0])
            and x['bbox'][1] > min(room['bbox'][0]+1, room['center'][0])
        ]
        if more_north_rooms == []:
            valid_north_rooms.append([room['region_id'], room['region_name']])
            
    all_x_max = [x['bbox'][1] for x in traj_room_info]
    most_max_x = max(all_x_max)
    east_rooms = [x for x in traj_room_info
                  if most_max_x-5 < x['bbox'][1] <= most_max_x]
    valid_east_rooms = []
    for room in east_rooms:
        more_east_rooms = [
            x for x in traj_room_info
            if x['center'][0] > room['center'][0]
            and x['bbox'][-1] > min(room['bbox'][-2]+1, room['center'][2])
            and x['bbox'][-2] < max(room['bbox'][-1]-1, room['center'][2]) 
        ]
        if more_east_rooms == []:
            valid_east_rooms.append([room['region_id'], room['region_name']])
            
    all_x_min = [x['bbox'][0] for x in traj_room_info]
    most_min_x = min(all_x_min)
    west_rooms = [x for x in traj_room_info
                  if most_min_x <= x['bbox'][0] < most_min_x+5]
    valid_west_rooms = []
    for room in west_rooms:
        more_west_rooms = [
            x for x in traj_room_info
            if x['center'][0] < room['center'][0]
            and x['bbox'][-1] > min(room['bbox'][-2]+1, room['center'][2])
            and x['bbox'][-2] < max(room['bbox'][-1]-1, room['center'][2])
        ]
        if more_west_rooms == []:
            valid_west_rooms.append([room['region_id'], room['region_name']])
    
    return {
        'NORTH': valid_north_rooms,
        'WEST': valid_west_rooms,
        'SOUTH': valid_south_rooms,
        'EAST': valid_east_rooms
    }
    
            
def spatial_orientation_rotation(input_dict, mode='test'):
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    scene_region_dists, all_regions = input_dict['scene_region_dists'], input_dict['all_regions'], #input_dict['scene_NSWE']
    
    episodes = []
    
    #* 当前面向北，去视频中最南边的xx房间拿xx东西
    floors = [x[0].split('_')[0] for x in room_set]
    floors = list(set(floors))
    if len(floors) != 1:
        print(room_set)
        return []
    floor = floors[0]
    floor_NSWE = get_nswe(all_regions[floor]['regions'], room_set)
    floor_all_rooms = sum(list(floor_NSWE.values()), [])
    
    #* 房间在所在方位是唯一的，不在角上
    room_set_names = [x[1] for x in room_set]
    valid_rooms = []  # [ (room_id, room_name, room_direction) ]
    for room in room_set:
        if room[1] in skip_rooms:
            continue
        if room_set_names.count(room[1]) < 2:
            continue
        room_direction = [k for k,v in floor_NSWE.items() if list(room) in v]
        if len(room_direction) != 1:
            continue
        room_direction = room_direction[0]
        direction_room_names = [x[1] for x in floor_NSWE[room_direction]]
        if direction_room_names.count(room[1]) > 1:
            continue
        if floor_all_rooms.count(room) > 1:
            continue
        valid_rooms.append([room[0], room[1], room_direction])
    
    for tgt_room in valid_rooms:
        region_idx = tgt_room[0].split('_')[-1]
        target_insts, target_cat = select_obj_gen_instr(region_idx, objects_dict)
        if target_cat is None:
            continue
        target_room_name = standard_room_name(tgt_room[1])
        tgt_direction = tgt_room[-1]
        frame_idxs = [idx for idx, room in enumerate(detailed_rooms)
                    if room!=[] and room[0]==tgt_room[0]]
        if frame_idxs == []:
            continue
        
        init_direction = [(i, direct) for i, direct in enumerate(DIRECTIONS) if direct!=tgt_direction]
        # if mode == 'test':
        #     cand_init_directs = [random.choice(init_direction)]
        # else:
        cand_init_directs = init_direction
            
        for init_direct in cand_init_directs:
            instruction = f"Now facing {init_direct[1].lower()}. Go to the {tgt_direction.lower()}ernmost {target_room_name} that was visited in the video. Locate the {target_cat}."
            print(instruction)
            
            cand_start_pos_room = [
                [traj_item['position'], list(room)]
                for traj_item, room in zip(ori_trajectory, ori_detailed_rooms)
                if room !=[] and room[0] != tgt_room[0]
                and list(room) in detailed_rooms
            ]
            if cand_start_pos_room == []:
                continue
            start_pos_room = random.choice(cand_start_pos_room)
            q = quat_from_angle_axis(math.radians(90*init_direct[0]), np.array([0.0, 1.0, 0.0]))
            
            episode = {
                'start_room': start_pos_room[1],
                'start_position': start_pos_room[0],
                'start_rotation': [q.x, q.y, q.z, q.w],
                'instruction': {
                    'task_type': 'room_spatial_orientation/rotation',
                    'instruction_text': instruction,
                    'instance_id': target_insts,
                },
                'object_category': target_cat,
                'frame_idxs': frame_idxs,
                'target_room': tgt_room[:2],
                'goals': get_ep_goals(target_insts, objects_dict, inst_viewpoints),
                'tag': (start_pos_room[1][0], init_direct[1].lower(), tgt_direction, tgt_room[0], input_dict['scan'])
            }
            episodes.append(episode)
    return episodes
    
    
    
def spatial_orientation_position(input_dict, mode='test'):
    ori_trajectory, ori_detailed_rooms, trajectory = input_dict['ori_trajectory'], input_dict['ori_detailed_rooms'], input_dict['trajectory']
    abs_rooms, detailed_rooms, room_set = input_dict['abs_rooms'], input_dict['detailed_rooms'], input_dict['room_set']
    objects_dict, inst_viewpoints, inst_instructions = input_dict['objects_dict'], input_dict['inst_viewpoints'], input_dict['inst_instructions']
    scene_region_dists, all_regions, scene_NSWE = input_dict['scene_region_dists'], input_dict['all_regions'], input_dict['scene_NSWE']
    
    episodes = []
    
    #* 视频中有3个不同的bedroom，初始化在房屋最东边，去西边的那个bedroom
    floors = [x[0].split('_')[0] for x in room_set]
    floors = list(set(floors))
    if len(floors) != 1:
        print(room_set)
        return []
    floor = floors[0]
    floor_NSWE = scene_NSWE[floor]
    floor_all_rooms = sum(list(floor_NSWE.values()), [])
    
    