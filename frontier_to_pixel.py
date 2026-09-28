import os
import argparse
import json
import gzip
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis

from src.habitat import make_simple_cfg
from src.geom import get_cam_intr

from scipy.spatial.transform import Rotation as R

# Pixel labels are projected into this image resolution.
IMG_SIZE = 384
REL_VIEW_DEG = [-180, -90, 90, 0]


def look_at_rotation(current_pos, target_pos, cam_height=1.5, max_pitch=np.pi / 3):
    x, y, z = current_pos
    direction = np.array(target_pos) - np.array([x, y + cam_height, z])
    direction = direction / np.linalg.norm(direction)
    x, y, z = direction

    pitch = np.arcsin(np.clip(y, -np.sin(max_pitch), np.sin(max_pitch)))
    hori_dir = np.array([x, 0, z])
    hori_dir = hori_dir / np.linalg.norm(hori_dir)

    quat_pitch = quat_from_angle_axis(pitch, np.array([1, 0, 0]))
    quat_yaw = quat_from_two_vectors(habitat_sim.geo.FRONT, hori_dir)
    final_quat = quat_yaw * quat_pitch
    return final_quat, quat_yaw, quat_pitch


def world_to_camera(points_world, cam_pos, cam_rot_quat):
    """Convert a world-frame point to the camera frame.

    cam_rot_quat is (x, y, z, w).
    """
    rot_mat_w2c = R.from_quat(cam_rot_quat).inv().as_matrix()
    return (rot_mat_w2c @ (points_world - cam_pos).T).T


def quaternion_to_yaw(quat):
    """Convert a Habitat quaternion [x, y, z, w] to yaw about +Y, in radians."""
    quat = np.array(quat, dtype=np.float32)
    if quat[-1] < 0:
        quat = quat * -1
    x, y, z, w = quat
    forward_x = 2 * (x * z + w * y)
    forward_z = 1 - 2 * (x * x + y * y)
    yaw = np.arctan2(forward_x, forward_z)
    return np.mod(yaw + np.pi, 2 * np.pi) - np.pi


def modify_deg(x):
    while x > 180:
        x -= 360
    while x < -180:
        x += 360
    return x


def per_episode(episode_dir, episode_name, K, sim, agent, goal_position):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    if not os.path.exists(os.path.join(episode_dir, "target.png")):
        return None
    data = json.load(open(os.path.join(episode_dir, "result.json"), "r"))
    if data == []:
        return None
    if data[-1]["goal_in_choice"] is False:
        return None

    result = []
    prev_rotation = None
    for step, info in enumerate(data):
        assert step == info["step"]
        agent_position = info["agent_position"]
        target_type, target_img = info["choice"]
        snapshots = info["snapshots"]

        if target_type == "frontier":
            target_frontier = info["frontiers"][target_img]
            target_position = target_frontier["position"]
            _, quat_yaw, _ = look_at_rotation(agent_position, target_position)
            frontier_yaw_deg = np.rad2deg(
                quaternion_to_yaw([quat_yaw.x, quat_yaw.y, quat_yaw.z, quat_yaw.w])
            )

            cur_key = f"{step}-view_3.png"
            if cur_key in snapshots:
                agent_rotation = snapshots[cur_key]["rotation"]
                cur_yaw_deg = np.rad2deg(quaternion_to_yaw(agent_rotation))
            else:
                step_keys = [x for x in snapshots if x.startswith(str(step))]
                if step_keys == [] and prev_rotation is None:
                    return None
                elif step_keys == []:
                    agent_rotation = [
                        prev_rotation.x, prev_rotation.y, prev_rotation.z, prev_rotation.w
                    ]
                    cur_yaw_deg = np.rad2deg(quaternion_to_yaw(agent_rotation))
                else:
                    view_idx = int(step_keys[0][:-4][-1])
                    view_rotation = snapshots[step_keys[0]]["rotation"]
                    view_yaw_deg = np.rad2deg(quaternion_to_yaw(view_rotation))
                    cur_yaw_deg = view_yaw_deg - REL_VIEW_DEG[view_idx]
            deg_dist = modify_deg(modify_deg(frontier_yaw_deg) - modify_deg(cur_yaw_deg))
            view_diff = [abs(deg_dist - x) for x in REL_VIEW_DEG]
            min_idx = np.argmin(view_diff)
            target_view_idx = min_idx
            snap_yaw_deg = modify_deg(cur_yaw_deg + REL_VIEW_DEG[min_idx])
            snap_rotation = quat_from_angle_axis(np.deg2rad(snap_yaw_deg), np.array([0, 1, 0]))

            agent_state = habitat_sim.AgentState()
            agent_state.position = agent_position
            agent_state.rotation = snap_rotation
            agent.set_state(agent_state)

            sensor_state = sim.get_agent(0).get_state().sensor_states["color_sensor"]
            cam_pos = sensor_state.position
            cam_rot = sensor_state.rotation
            cam_rot = np.array(cam_rot.imag.tolist() + [cam_rot.real])

            target_cam = world_to_camera(np.array(target_position), cam_pos, cam_rot)
            u = int(fx * target_cam[0] / target_cam[2] * (-1) + cx)
            v = int(fy * target_cam[1] / target_cam[2] + cy)
            if u > IMG_SIZE or v > IMG_SIZE:
                print(episode_name)
                return None

            result.append({
                "step": step,
                "view_idx": int(target_view_idx),
                "pixel": [u, v],
            })
            prev_rotation = quat_yaw
        else:
            snap_rotation = snapshots[target_img]["rotation"]
            agent_state = habitat_sim.AgentState()
            agent_state.position = agent_position
            agent_state.rotation = snap_rotation
            agent.set_state(agent_state)

            sensor_state = sim.get_agent(0).get_state().sensor_states["color_sensor"]
            cam_pos = sensor_state.position
            cam_rot = sensor_state.rotation
            cam_rot = np.array(cam_rot.imag.tolist() + [cam_rot.real])

            target_cam = world_to_camera(np.array(goal_position), cam_pos, cam_rot)
            u = int(fx * target_cam[0] / target_cam[2] * (-1) + cx)
            v = int(fy * target_cam[1] / target_cam[2] + cy)
            result.append({
                "step": step,
                "view_idx": int(target_img[:-4][-1]),
                "pixel": [u, v],
            })
    return result


def main(cfg, frontier_dir, instr_path_list, output_dir):
    cam_intrinsic = get_cam_intr(cfg.hfov, IMG_SIZE, IMG_SIZE)
    mp3d_dir = cfg.mp3d_dir

    episodes = []
    for path in instr_path_list:
        with gzip.open(path, "r") as f:
            episodes += json.load(f)["episodes"]
    episodes = {str(x["episode_id"]): x for x in episodes}

    episode_names = [
        name for name in os.listdir(frontier_dir)
        if os.path.isdir(os.path.join(frontier_dir, name)) and "_ep_" in name
    ]
    scans = sorted(set(name.split("_ep_")[0] for name in episode_names))
    os.makedirs(output_dir, exist_ok=True)

    for scan in scans:
        try:
            sim.close()
        except Exception:
            pass
        scene_mesh = os.path.join(mp3d_dir, scan, scan + ".glb")
        navmesh = os.path.join(mp3d_dir, scan, scan + ".navmesh")
        sim_settings = {
            "scene": scene_mesh,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": IMG_SIZE,
            "height": IMG_SIZE,
            "hfov": cfg.hfov,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        sim = habitat_sim.Simulator(make_simple_cfg(sim_settings))
        sim.pathfinder.load_nav_mesh(navmesh)
        agent = sim.initialize_agent(sim_settings["default_agent"])

        results = {}
        scan_episodes = [name for name in episode_names if name.startswith(scan + "_ep_")]
        for episode_name in tqdm(scan_episodes, desc=scan):
            episode_id = episode_name.split("_ep_")[-1]
            goal_position = episodes[episode_id]["goals"][0]["position"]
            pixel_result = per_episode(
                os.path.join(frontier_dir, episode_name),
                episode_name,
                cam_intrinsic,
                sim,
                agent,
                goal_position,
            )
            if pixel_result is None:
                continue
            results[episode_name] = pixel_result

        with open(os.path.join(output_dir, scan + ".json"), "w") as f:
            json.dump(results, f)


def merge_reps(output_root):
    results = {}
    total = 0
    for rep in os.listdir(output_root):
        rep_dir = os.path.join(output_root, rep)
        if not os.path.isdir(rep_dir):
            continue
        files = [name for name in os.listdir(rep_dir) if name.endswith(".json")]
        for file in tqdm(files, desc=rep):
            data = json.load(open(os.path.join(rep_dir, file), "r"))
            total += len(data)
            for ori_id, value in data.items():
                scene_id, episode_id = ori_id.split("_ep_")
                new_id = f"{scene_id}_ep_{rep[-1]}_{episode_id}"
                results[new_id] = value
    out_path = os.path.join(output_root, "fep_pixel_labels.json")
    with open(out_path, "w") as f:
        print(total, len(results))
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="config/collect_data.yaml", type=str)
    parser.add_argument("--frontier_dir", default="", type=str)
    parser.add_argument("--instruction", action="append", default=[], type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output_root", default="", type=str)
    args = parser.parse_args()

    if args.merge:
        merge_reps(args.output_root)
    else:
        cfg = OmegaConf.load(args.cfg_file)
        OmegaConf.resolve(cfg)
        main(cfg, args.frontier_dir, args.instruction, args.output_dir)
