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

from src.habitat import pose_habitat_to_tsdf, get_points_from_multigoal
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, SnapShot
from src.scene_memo import Scene
from src.utils import get_pts_angle_goatbench
from loggers.logger_collect import Logger


def main(cfg):
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
    file_path = cfg.file_path
    with gzip.open(file_path, 'r') as f:
        episodes = json.load(f)['episodes']
    num_episode = len(episodes)
    logging.info(
        f"Total number of episodes: {num_episode}"
    )
    
    # Initialize the logger
    logger = Logger(
        cfg.output_dir, voxel_size=cfg.tsdf_grid_size
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
            
            goal_instance = episode['instruction']['instance_id'][0]
            goal_position = episode['goals'][0]['position']

            pts, angle = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )

            # load scene
            try:
                del scene
            except:
                pass
            scene = Scene(scene_id, cfg, cfg_cg,)

            # initialize the TSDF
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
            )

            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )
            logger.init_task(pts, tsdf_planner)
            logging.info(f"\n\nScene {scene_id} initialization successful!")
            
            # run steps
            cnt_step = -1

            # reset tsdf planner
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None
            choice = None

            save_result = []
            while cnt_step < num_step - 1:
                cnt_step += 1
                logging.info(
                    f"\n== step: {cnt_step} =="
                )
                dist, _ = get_points_from_multigoal(pts, episode['goals'], scene.pathfinder)
                step_logdir = {
                    'step': cnt_step,
                    'agent_position': list(pts),
                    'explore_dist': logger.explore_dist,
                    'dist_to_goal': dist,
                    'snapshots': {}, # {image: {position, rotation, objids} }
                    'frontiers': {}, # {image: {frontier_id, position, orientation} }
                    'choice': [],  # e.g. ['frontier', <imgid>]  
                    'goal_in_choice': False
                }
                if len(save_result) > 4 \
                    and save_result[-4]['dist_to_goal'] < dist \
                    and save_result[-3]['dist_to_goal'] < dist \
                    and save_result[-2]['dist_to_goal'] < dist \
                    and save_result[-1]['dist_to_goal'] < dist:
                    logging.info("-------------- Warning! Getting further from the goal!")
                    break
                    

                #! (1) Observe the surroundings, update the scene graph and occupancy map
                angle_increment = 60 * np.pi / 180 
                total_views = 6
                all_angles = [
                    angle + angle_increment * (i - total_views // 2)
                    for i in range(total_views)
                ]
                # Let the main viewing angle be the last one to avoid potential overwriting problems
                main_angle = all_angles.pop(total_views // 2)
                all_angles.append(main_angle)

                rgb_egocentric_views = []
                all_added_obj_ids = ( 
                    []
                )  # Record all the objects that are newly added in this step
                for view_idx, ang in enumerate(all_angles):
                    # For each view
                    obs, cam_pose = scene.get_observation(pts, angle=ang)
                    rgb = obs["color_sensor"]
                    depth = obs["depth_sensor"]
                    
                    clean_tsdf = TSDFPlanner(vol_bnds=tsdf_bnds, voxel_size=cfg.tsdf_grid_size,
                        floor_height=floor_height, floor_height_offset=0,
                        pts_init=pts, init_clearance=cfg.init_clearance * 2, save_visualization=cfg.save_visualization,
                    )
                    
                    # Update depth map, occupancy map
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
                    clean_tsdf.integrate(
                        color_im=rgb,
                        depth_im=depth,
                        cam_intr=cam_intr,
                        cam_pose=pose_habitat_to_tsdf(cam_pose),
                        obs_weight=1.0,
                        margin_h=int(cfg.margin_h_ratio * img_height),
                        margin_w=int(cfg.margin_w_ratio * img_width),
                        explored_depth=cfg.explored_depth,
                    )

                    # collect all view features
                    obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                    added_obj_ids = scene.update_scene_graph(
                        pts, clean_tsdf, img_path=obs_file_name
                    )
                    scene.all_observations[obs_file_name] = rgb
                    rgb_egocentric_views.append(rgb)
                    plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)  # save current observations
                    all_added_obj_ids += added_obj_ids
                    
                #! (2) Update Memory Snapshots with hierarchical clustering
                # Choose all the newly added objects as well as the objects nearby as the cluster targets
                all_added_obj_ids = [
                    obj_id
                    for obj_id in all_added_obj_ids
                    if obj_id in scene.objects
                ]
                for obj_id, obj in scene.objects.items():
                    if (
                        np.linalg.norm(np.array(obj["bbox"]['center'])[[0, 2]] - pts[[0, 2]])
                        < cfg.scene_graph.obj_include_dist + 0.5
                    ):
                        all_added_obj_ids.append(obj_id)
                scene.update_snapshots(
                    obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
                )
                for imgid, snap in scene.snapshots.items():
                    step_logdir['snapshots'][imgid] = {
                        'position': snap.agent_position.tolist(),
                        'rotation': snap.agent_rotation,
                        'objids': list(snap.full_obj_list)
                    }
                    
                logging.info(
                    f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
                )

                #! (3) Update the Frontier Snapshots
                update_success = tsdf_planner.update_frontier_map(
                    pts=pts,
                    cfg=cfg.planner,
                    scene=scene,
                    cnt_step=cnt_step,
                    save_frontier_image=cfg.save_visualization,
                    eps_frontier_dir=eps_frontier_dir,
                    prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                )
                if not update_success:
                    logging.info("Warning! Update frontier map failed!")
                for frontier in tsdf_planner.frontiers:
                    step_logdir['frontiers'][frontier.image] = {
                        'position': tsdf_planner.voxel2habitat(frontier.position.astype(int)).tolist(),
                        'orientation': frontier.orientation.tolist(),
                    }

                #! (4) select according to shortest path
                if goal_instance in scene.objects:
                    snapshot = [v for k,v in scene.snapshots.items()
                                if goal_instance in v.cluster][0]
                    update_success = tsdf_planner.set_snapshot_as_next_point(
                        cfg.planner, pts, snapshot, scene
                    )
                    if not update_success:
                        logging.info("Invalid: set_snapshot_as_next_point failed!")
                        break
                    step_logdir['choice'] = ['snapshot', tsdf_planner.max_point.image]
                else:
                    if tsdf_planner.frontiers == []:
                        logging.info("Invalid: No frontiers to choose from ! ! !")
                        break
                    update_success = tsdf_planner.choose_frontier_and_set_next_point(
                        pts, cfg.planner, scene, episode['goals']
                    )
                    if not update_success:
                        logging.info("Invalid: choose_frontier_and_set_next_point failed!")
                        break
                    step_logdir['choice'] = ['frontier', tsdf_planner.max_point.image]
                    
                choice = tsdf_planner.max_point
                if isinstance(choice, SnapShot):
                    step_logdir['goal_in_choice'] = goal_instance in choice.full_obj_list
                save_result.append(step_logdir)

                #! (5) Agent navigate to the target point
                return_values = tsdf_planner.agent_step(
                    pts=pts,
                    angle=angle,
                    objects=scene.objects,
                    snapshots=scene.snapshots,
                    pathfinder=scene.pathfinder,
                    cfg=cfg.planner,
                    path_points=None,
                    save_visualization=cfg.save_visualization,
                )
                if return_values[0] is None:
                    logging.info(
                        f"Invalid: agent_step failed!"
                    )
                    break

                # update agent's position and rotation
                pts, angle, pts_voxel, fig, _, target_arrived = return_values
                logger.log_step(pts_voxel=pts_voxel)
                logging.info(
                    f"Current position: {pts}, {logger.explore_dist:.3f}, Target position: {goal_position}"
                )

                # sanity check about objects, scene graph, snapshots, ...
                scene.sanity_check(cfg=cfg)

                if cfg.save_visualization:
                    # save the top-down visualization
                    goal_positon = episode['goals'][0]['position']
                    logger.save_topdown_visualization(
                        global_step=cnt_step,
                        goal_pos_voxel = tsdf_planner.habitat2voxel(goal_positon),
                        goal_observed = goal_instance in scene.objects,
                        fig=fig,
                    )
                    # save the visualization of vlm's choice at each step
                    try:
                        instr_text = episode['instruction']['instruction_text']
                    except:
                        instr_text = "inst_img_goal"
                    caption = f"{episode['instruction']['task_type']}\n{instr_text}\n{goal_instance}"
                    caption += f"\nSnapshots: {', '.join(list(scene.snapshots))}"
                    caption += f"\nSelection: {': '.join(step_logdir['choice'])}"
                    logger.save_frontier_visualization(
                        global_step=cnt_step,
                        tsdf_planner=tsdf_planner,
                        max_point_choice=choice,
                        global_caption=caption,
                    )

                #! (6) Check if the agent has arrived at the target to finish the question
                if type(choice) == SnapShot and target_arrived:
                    # when the target is a snapshot, and the agent arrives at the target
                    # we consider the task is finished, take an observation and save the chosen target snapshot
                    obs, _ = scene.get_observation(pts, angle=angle)
                    rgb = obs["color_sensor"]
                    plt.imsave(
                        os.path.join(
                            logger.episode_dir, f"target.png"
                        ),
                        rgb,
                    )
                    break

            # save the results at the end of each episode
            logger.save_result(save_result)

            logging.info(f"Episode {episode_id} finish")
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    parser.add_argument("--exp_name", default='', type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    if args.exp_name != '':
        cfg.exp_name = args.exp_name
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
