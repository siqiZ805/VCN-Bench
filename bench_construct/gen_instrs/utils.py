import math
import random
import numpy as np
from collections import Counter
from habitat_sim.utils.common import quat_from_angle_axis
from constants import *

def get_ep_goals(target_insts, objects_dict, inst_viewpoints):
    goals = []
    for inst in target_insts:
        inst_info = objects_dict[inst]
        min_points = inst_info['min_points']
        max_points = inst_info['max_points']
        view_points = []
        for point in inst_viewpoints[inst]['view_points']:
            view_points.append({
                'agent_state': { 'position': point }
            })
        goals.append({
            'position': inst_info['position'],
            'bbox': [*min_points[:2], max_points[2], *max_points[:2], min_points[2]],
            'view_points': view_points
        })
    return goals
        

def abstract_traj(rooms_along_traj):
    abstract_rooms = []
    for room in rooms_along_traj:
        if room == []:
            continue
        
        if abstract_rooms == []:
            abstract_rooms.append(room)
        else:
            if room[0] != abstract_rooms[-1][0]:
                abstract_rooms.append(room)
    return abstract_rooms

def filt_small_objs(instids, objects_dict, thres=0.5):
    filtered_instances = []
    for inst in instids:
        info = objects_dict[inst]
        minp = info['min_points']
        maxp = info['max_points']
        c = maxp[0]-minp[0] + maxp[1]-minp[1] + minp[2]-maxp[2]
        if c > thres:
            filtered_instances.append(inst)
    return filtered_instances

def select_obj_gen_instr(
    region_idx, object_dict, do_filt=True
):
    region_instances = {
        k:v for k,v in object_dict.items() 
        if f"region{region_idx}/" in k  
        and v['category'] not in skip_objects 
    }
    if len(region_instances) == 0:
        return None, None
    
    if do_filt:
        filt_instids = filt_small_objs(list(region_instances), object_dict)
        inst_categories = [v['category'] for k,v in region_instances.items()
                        if k in filt_instids]
    else:
        inst_categories = [v['category'] for k,v in region_instances.items()]
    if len(inst_categories) == 0:
            return None, None
        
    inst_cat_count = Counter(inst_categories)
    single_cats = [
        cat for cat, count in inst_cat_count.items()
        if count == 1
    ]
    
    r = random.random()
    if (len(single_cats)==1 and r<=0.5) or (len(single_cats)==2 and r<=0.7) or (len(single_cats)>2 and r<=0.9):
        target_cat = random.choice(single_cats)
    else:
        target_cat = random.choice(list(set(inst_categories)))
        
    target_instance = [k for k,v in region_instances.items() if v['category']==target_cat]
    return target_instance, target_cat
          

def select_inst_gen_instr(
    region_idx, object_dict, inst_instructions, 
    do_filt=False, my_skip_objects=None
):
    if my_skip_objects is None:
        my_skip_objects = skip_objects
    region_instances = {k:v for k,v in object_dict.items() 
                        if f"region{region_idx}/" in k  
                        and v['category'] not in my_skip_objects 
                        and k in inst_instructions}
    if len(region_instances) == 0:
        return None, None
    
    if do_filt:
        filt_instids = filt_small_objs(list(region_instances), object_dict)
        inst_categories = [v['category'] for k,v in region_instances.items()
                        if k in filt_instids]
        if len(inst_categories) == 0:
            return None, None
    else:
        inst_categories = [v['category'] for k,v in region_instances.items()]
    
    inst_cat_count = Counter(inst_categories)
    single_cats = [
        cat for cat, count in inst_cat_count.items() 
        if count == 1 and cat != 'object'
    ]
    r = random.random()
    
    inst_instr = ''
    # if single_cats != []:
    if (len(single_cats)==1 and r<=0.5) or (len(single_cats)==2 and r<=0.7) or (len(single_cats)>2 and r<=0.9):
        target_cat = random.choice(single_cats)
        target_instance = [k for k,v in region_instances.items() if v['category']==target_cat][0]
    else:
        target_instance = random.choice(list(region_instances.keys()))
        # target_cat = target_inst_info['category']
        
    target_inst_info = region_instances[target_instance]
    target_cat = target_inst_info['category']
    if inst_cat_count[target_cat] == 1:
        if target_cat == 'object':
            inst_instr = inst_instructions[target_instance]
        else:
            inst_instr = f"Locate the {target_cat.lower()}."
    else:  # e.g. 不止一个chair
        confuse_insts = {k:v for k,v in region_instances.items()
                          if v['category'] == target_cat}
        confuse_feats = [
            (
                v['unique_description']['fine grained category'],
                v['common']['color']
            )
            for v in confuse_insts.values()
        ]
        target_fg_cat = target_inst_info['unique_description']['fine grained category']
        if target_fg_cat not in [x[0] for x in confuse_feats]:
            inst_instr = f"Locate the {target_fg_cat.lower()}."
        target_color = target_inst_info['common']['color']
        if target_color not in [x[1] for x in confuse_feats]:
            inst_instr = f"Locate the {target_color.lower()} {target_cat.lower()}."
        
        if inst_instr == '':
            inst_instr = inst_instructions[target_instance]
            
    return target_instance, inst_instr

def standard_room_name(room_name):
    if room_name in mp3droom_to_myroom:
        my_room_name = mp3droom_to_myroom[room_name]
        name_list = my_room_name.split('|')
        room_name = random.choice(name_list)
    return room_name


def random_yaw_quaternion_list():
    q = quat_from_angle_axis(math.radians(random.randint(0, 360)), np.array([0.0, 1.0, 0.0]))
    return [q.x, q.y, q.z, q.w]