from src.runtime import configure_environment, run_cli, select_episode_range

configure_environment()


def main(cfg, start_ratio=0.0, end_ratio=1.0, split=1):
    """Run navigation evaluation."""
    import gzip
    import json
    import logging
    import math
    import os
    import random

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf

    from model.query_dualvln import build_model, query_qwen
    from src.geom import get_cam_intr, get_scene_bnds
    from src.habitat import pos_habitat_to_normal, pose_habitat_to_tsdf
    from src.logger import Logger
    from src.scene_memo import Scene
    from src.tour_video import SAMPLING_PROTOCOL, sample_tour_video
    from src.tsdf_planner import SnapShot, TSDFPlanner
    from src.utils import calc_agent_subtask_distance, get_pts_angle_goatbench

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
    scenes = [x for x in os.listdir(MP3D_SCENE) if "." not in x]

    # test data
    file_path = cfg.test_data_path
    with gzip.open(file_path, "r") as f:
        episodes = select_episode_range(
            json.load(f)["episodes"], start_ratio, end_ratio
        )
    num_episode = len(episodes)
    logging.info(f"Total number of episodes: {num_episode}")

    # model
    model, processor = build_model(cfg.qwen_path, cfg.qwen_processor_path)
    logging.info(f"Load QWEN model successful!")

    # Initialize the logger
    logger = Logger(cfg.output_dir, voxel_size=cfg.tsdf_grid_size)

    scene = None
    for scene_id in scenes:
        scene_data = [x for x in episodes if scene_id in x["scene_id"]]
        if scene_data == []:
            continue
        total_episodes = len(scene_data)

        for episode_idx, episode in enumerate(scene_data):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            output_path = os.path.join(
                cfg.output_dir, f"{scene_id}_ep_{episode_id}", "result.json"
            )
            if os.path.exists(output_path):
                continue

            if scene is not None:
                scene.close()
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
            )

            pts_init, angle_init = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )
            floor_height = pts_init[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)

            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )

            # load prior video
            hkey = episode["hkey"]
            video_dir = os.path.join(cfg.video_root, scene_id, hkey)
            all_rgbs = sorted(os.listdir(video_dir))
            sidx = episode["start_idx"]
            eidx = episode["end_idx"]
            tour_video = all_rgbs[sidx:eidx]
            tour_video = sample_tour_video(tour_video, cfg.video_max_frame)

            # nav goal
            _instr = episode["instruction"]
            if "image" in _instr["task_type"]:
                goal = _instr["img_path"]
            else:
                goal = _instr["instruction_text"]

            max_attempts = 3
            for attempt in range(max_attempts):
                logging.info(
                    f"Episode {episode_id} attempt {attempt + 1}/{max_attempts}"
                )
                pts = np.array(pts_init, copy=True)
                angle = float(angle_init)
                for artifact_dir in (eps_frontier_dir, eps_snapshot_dir):
                    for name in os.listdir(artifact_dir):
                        artifact_path = os.path.join(artifact_dir, name)
                        if os.path.isfile(artifact_path):
                            os.remove(artifact_path)
                scene.clear_up_detections()

                tsdf_planner = TSDFPlanner(
                    vol_bnds=tsdf_bnds,
                    voxel_size=cfg.tsdf_grid_size,
                    floor_height=floor_height,
                    floor_height_offset=0,
                    pts_init=pts,
                    init_clearance=cfg.init_clearance * 2,
                    save_visualization=cfg.save_visualization,
                )  # just for functions
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                metadata = logger.init_eval_task(episode, pts, tsdf_planner)
                logging.info(f"\n\nScene {scene_id} initialization successful!")

                # run steps
                global_step = -1
                save_result = {
                    "tour_video_sampling": SAMPLING_PROTOCOL,
                    "video_max_frame": cfg.video_max_frame,
                    "prior_video_frames": tour_video,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                }
                init_rgbs = []
                hist_rgbs = []  # each element is [name, rgb]
                while global_step < num_step - 1:
                    global_step += 1
                    logging.info(f"\n== global step: {global_step} ==")

                    # 1. Observe the surroundings, update the scene graph
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
                        depth = obs["depth_sensor"]
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
                            image=obs_file_name,
                            color=(random.random(), random.random(), random.random()),
                            obs_point=tsdf_planner.habitat2voxel(pts),
                            agent_position=pts,
                            agent_rotation=rotation,
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

                    if global_step == 0:
                        init_rgbs = list(rgb_egocentric_views)
                    vlm_output_dict = query_qwen(
                        model,
                        processor,
                        tour_video,
                        hist_rgbs,
                        init_rgbs,
                        rgb_egocentric_views,
                        video_dir,
                        cfg.video_root,
                        goal,
                    )
                    if vlm_output_dict is None:
                        break
                    save_result[f"step_{global_step}"] = {
                        "hist_rgbs": [x[0] for x in hist_rgbs],
                        "goal": goal,
                        "vlm_output": vlm_output_dict,
                        "position": list(pts),
                    }

                    pred_view_idx = vlm_output_dict["view_idx"]
                    if pred_view_idx < 0 or pred_view_idx > 3:
                        break

                    if pred_view_idx <= 2:
                        add_idx = [x for x in range(pred_view_idx + 1)]
                    elif pred_view_idx == 3:
                        add_idx = [0, 3]
                    for idx in add_idx:
                        hist_rgbs.append(
                            [f"{global_step}-view_{idx}.png", rgb_egocentric_views[idx]]
                        )

                    depth = depth_list[pred_view_idx]
                    cam_pos, cam_rot = (
                        cam_pos_list[pred_view_idx],
                        cam_rot_list[pred_view_idx],
                    )
                    u, v = vlm_output_dict["pixel"]
                    if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                        target_position = scene.pixel_to_point(
                            (u, v), depth, cam_pos, cam_rot
                        )
                    else:
                        target_position = None
                        break

                    vis_img = np.ascontiguousarray(hist_rgbs[-1][-1][:, :, ::-1])
                    cv2.circle(vis_img, (u, v), 10, (0, 0, 255))
                    cv2.imwrite(
                        os.path.join(eps_frontier_dir, hist_rgbs[-1][0]), vis_img
                    )

                    voxel_position = tsdf_planner.normal2voxel(
                        pos_habitat_to_normal(target_position)
                    )
                    tsdf_planner.target_point = voxel_position[:2]
                    return_values = tsdf_planner.agent_step_eval(
                        pts=pts,
                        angle=angle,
                        pathfinder=scene.pathfinder,
                        cfg=cfg.planner,
                    )
                    if return_values[0] is None:
                        logging.info(f"Agent_step failed!")
                        break

                    # update agent's position and rotation
                    pts, angle, pts_voxel, target_arrived = return_values
                    logger.log_step(pts_voxel=pts_voxel)
                    logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")

                    if vlm_output_dict["stop"] == "true":
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
                if success_by_distance:
                    break
                if attempt + 1 < max_attempts:
                    logging.info(
                        f"Retry episode {episode_id}: attempt {attempt + 1} failed at distance {distance}"
                    )
            pl = metadata["gt_dist"] / max(metadata["gt_dist"], logger.explore_dist)
            save_result.update(
                {
                    "final_position": list(pts),
                    "success_by_distance": success_by_distance,
                    "dist_to_goal": distance,
                    "nav_dist": logger.explore_dist,
                    "gt_dist": metadata["gt_dist"],
                    "spl_by_distance": success_by_distance * pl,
                }
            )
            logger.save_result(save_result)
            logging.info(f"Episode {episode_id} finish")

    if scene is not None:
        scene.close()


if __name__ == "__main__":
    run_cli(main, "config/eval_mvdualvln.yaml")
