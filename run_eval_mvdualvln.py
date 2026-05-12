import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import math
import time
import json
import gzip
import logging
import matplotlib.pyplot as plt
import cv2

from src.habitat import pose_habitat_to_tsdf, pos_habitat_to_normal
from src.geom import get_cam_intr, get_scene_bnds
from src.scene_memo import Scene
from src.tsdf_planner import TSDFPlanner, SnapShot
from src.utils import calc_agent_subtask_distance, get_pts_angle_goatbench
from src.logger import Logger
from model.query_dualvln import query_qwen, build_model


def main(cfg, start_ratio=0.0, end_ratio=1.0, split=1):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    MP3D_SCENE = cfg.mp3d_dir
    scenes = [x for x in os.listdir(MP3D_SCENE) if '.' not in x]
    
    # test data
    file_path = cfg.test_data_path
    with gzip.open(file_path, 'r') as f:
        episodes = json.load(f)['episodes']
    num_episode = len(episodes)
    logging.info(f"Total number of episodes: {num_episode}")
    
    # model
    model, processor = build_model(cfg.qwen_path)
    logging.info(f"Load QWEN model successful!")
    
    # Initialize the logger
    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
    )
    
    for scene_id in scenes:
        scene_data = [x for x in episodes if scene_id in x['scene_id']]
        if scene_data == []:
            continue
        total_episodes = len(scene_data)
        
        for episode_idx, episode in enumerate(scene_data):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]
            
            if os.path.exists(os.path.join(cfg.output_dir, f"{scene_id}_ep_{episode_id}", 'result.json')):
                continue
  
            try:
                del scene
            except:
                pass
            scene = Scene(scene_id, cfg, cfg_cg,)
            
            pts, angle = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)
            
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            ) #? just for functions
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None
            
            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )
            metadata = logger.init_memo_task(episode, pts, tsdf_planner)
            logger.init_task_my(pts, tsdf_planner)
            logging.info(f"\n\nScene {scene_id} initialization successful!")
            
            # load prior video
            hkey = episode['hkey']
            video_dir = os.path.join(cfg.video_root, scene_id, hkey)
            all_rgbs = sorted(os.listdir(video_dir))
            sidx = episode['start_idx']
            eidx = episode['end_idx']
            tour_video = all_rgbs[sidx : eidx]
            # downsample
            video_len = eidx - sidx
            step = 1
            while video_len / step >= cfg.video_max_frame:
                step += 1
            tour_video = tour_video[::step]

            # nav goal
            _instr = episode['instruction']
            if 'image' in _instr['task_type']:
                goal = _instr['img_path']
            else:
                goal = _instr['instruction_text']
            
            # run steps
            global_step = -1
            save_result = {}
            hist_rgbs = []  # each element is [name, rgb]
            while global_step < num_step - 1:
                global_step += 1
                logging.info(
                    f"\n== global step: {global_step} =="
                )
                
                #* 1. Observe the surroundings, update the scene graph
                angle_increment = 90 * np.pi / 180 
                total_views = 4
                all_angles = [
                    angle + angle_increment * (i - total_views // 2)
                    for i in range(total_views)
                ]
                # clockwise order
                all_angles = all_angles[2:] + all_angles[:2]
                
                rgb_egocentric_views = []
                depth_list, cam_pos_list, cam_rot_list = [], [], []
                for view_idx, ang in enumerate(all_angles):
                    obs, cam_pose = scene.get_observation(pts, angle=ang)
                    rgb = obs["color_sensor"]
                    depth = obs['depth_sensor']
                    rgb_egocentric_views.append(rgb)
                    depth_list.append(depth)
                    cam_pos, cam_rot = scene.get_cam_pos_rot()
                    cam_pos_list.append(cam_pos)
                    cam_rot_list.append(cam_rot)
                    
                    # update scene frames
                    obs_file_name = f"{global_step}-view_{view_idx}.png"
                    rotation = scene.agent.get_state().rotation
                    rotation = rotation.imag.tolist() + [rotation.real]
                    frame = SnapShot(
                        image=obs_file_name, color=(random.random(), random.random(), random.random()),
                        obs_point=tsdf_planner.habitat2voxel(pts),
                        agent_position=pts, agent_rotation=rotation
                    )
                    scene.frames[obs_file_name] = frame
                    scene.all_observations[obs_file_name] = rgb
                    
                    plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)
                    
                    tsdf_planner.integrate(
                        color_im=rgb,
                        depth_im=depth,
                        cam_intr=cam_intr,
                        cam_pose=pose_habitat_to_tsdf(cam_pose),
                        obs_weight=1.0,
                        margin_h=int(cfg.margin_h_ratio * img_height),
                        margin_w=int(cfg.margin_w_ratio * img_width),
                        explored_depth=cfg.explored_depth,
                    )
                    
                vlm_output_dict = query_qwen(
                    scene, cfg, global_step,
                    model, processor, 
                    tour_video, hist_rgbs, rgb_egocentric_views,
                    video_dir, cfg.video_root, goal,
                )
                if vlm_output_dict is None:
                    break
                save_result[f"step_{global_step}"] = {
                    'hist_rgbs': [x[0] for x in hist_rgbs],
                    'goal': goal,
                    'vlm_output': vlm_output_dict,
                    'position': list(pts)
                }
                
                pred_view_idx = vlm_output_dict['view_idx']
                if pred_view_idx < 0 or pred_view_idx > 3:
                    break
                
                if pred_view_idx <=2:
                    add_idx = [x for x in range(pred_view_idx+1)]
                elif pred_view_idx == 3:
                    add_idx = [0, 3]
                for idx in add_idx:
                    hist_rgbs.append([f"{global_step}-view_{idx}.png", rgb_egocentric_views[idx]])
                    
                depth = depth_list[pred_view_idx]
                cam_pos, cam_rot = cam_pos_list[pred_view_idx], cam_rot_list[pred_view_idx]
                u, v = vlm_output_dict['pixel']
                if 0<=u<=384 and 0<=v<=384:
                    target_position = scene.pixel_to_point((u,v), depth, cam_pos, cam_rot)
                else:
                    target_position = None
                    break
                
                vis_img = np.ascontiguousarray(hist_rgbs[-1][-1][:,:,::-1])
                cv2.circle(vis_img, (u,v), 10, (0,0,255))
                cv2.imwrite(os.path.join(eps_frontier_dir, hist_rgbs[-1][0]), vis_img)
                
                voxel_position = tsdf_planner.normal2voxel(pos_habitat_to_normal(target_position))
                tsdf_planner.target_point = voxel_position[:2]
                return_values = tsdf_planner.agent_step_eval(
                    pts=pts, angle=angle, 
                    pathfinder=scene.pathfinder, cfg=cfg.planner,
                )
                if return_values[0] is None:
                    logging.info(
                        f"Agent_step failed!"
                    )
                    break
                
                # update agent's position and rotation
                pts, angle, pts_voxel, target_arrived = return_values
                logger.log_step(pts_voxel=pts_voxel)
                logging.info(
                    f"Current position: {pts}, {logger.explore_dist:.3f}"
                )
                
                if vlm_output_dict['stop'] == 'true':
                    break
                
            distance = calc_agent_subtask_distance(
                pts, metadata["viewpoints"], scene.pathfinder
            )
            if distance < cfg.success_distance:
                success_by_distance = True
                logging.info(
                    f"Success: agent reached the target viewpoint at distance {distance}!"
                )
            else:
                success_by_distance = False
                logging.info(
                    f"Fail: agent failed to reach the target viewpoint at distance {distance}!"
                )
            pl = metadata['gt_dist'] / max(metadata['gt_dist'], logger.explore_dist)
            save_result.update({
                'final_position': list(pts),
                'success_by_distance': success_by_distance,
                'dist_to_goal': distance,
                'nav_dist': logger.explore_dist,
                'gt_dist': metadata['gt_dist'],
                'spl_by_distance': success_by_distance * pl
            })
            logger.save_result(save_result)
            logging.info(f"Episode {episode_id} finish")
                    
                    
if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log",
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=args.split)