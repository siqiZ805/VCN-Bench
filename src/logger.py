import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from typing import Union

from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot


class Logger:
    def __init__(
        self, 
        output_dir, 
        voxel_size,  # used for calculating the moving distance
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        self.success, self.spl = {}, {}
        self.explore_dist = 0.0
        
    def init_episode(self, episode_id):
        self.episode_dir = os.path.join(self.output_dir, episode_id)
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")
        eps_snapshot_dir = os.path.join(self.episode_dir, "snapshot")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)
        os.makedirs(eps_snapshot_dir, exist_ok=True)

        return self.episode_dir, eps_frontier_dir, eps_snapshot_dir
    
    def init_collect_task(self, pts, tsdf_planner):
        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack(
            [self.pts_voxels, tsdf_planner.habitat2voxel(pts)[:2]]
        )
        self.explore_dist = 0.0
        
    def init_eval_task(self, episode, pts, tsdf_planner):
        #* 1. nav goal
        goal_category = episode['object_category']
        goal_obj_ids = episode['instruction']['instance_id']
        goal_positions = [episode['goals'][0]['position']]
        goal_positions_voxel = [tsdf_planner.habitat2voxel(p) for p in goal_positions]
        viewpoints = [
            vp['agent_state']['position']
            for vp in episode['goals'][0]['view_points']
        ]
        gt_dist = episode['dist_to_goal'] + 1e-6
        
        _instr = episode['instruction']
        if 'image' in _instr['task_type']:
            goal = _instr['img_path']
        else:
            goal = _instr['instruction_text']
            
        #* 2. prepare metadata
        metadata = {
            "goal": goal,
            "class": goal_category,
            "goal_obj_ids": goal_obj_ids,  # e.g. ["1mp3d_0070_region1/washbasin_10"]
            "goal_positions_voxel": goal_positions_voxel,  # also a list of positions for possible multiple objects
            "viewpoints": viewpoints,
            "gt_dist": gt_dist,
            "gt_bbox": episode['goals'][0]['bbox']
        }
        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack(
            [self.pts_voxels, tsdf_planner.habitat2voxel(pts)[:2]]
        )
        self.explore_dist = 0.0
        return metadata
        
    def log_step(self, pts_voxel):
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        self.explore_dist += (
            np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
        )
        
    def save_topdown_visualization(
        self, global_step, goal_pos_voxel, goal_observed, fig
    ):
        assert self.episode_dir is not None
        visualization_path = os.path.join(self.episode_dir, "visualization")
        os.makedirs(visualization_path, exist_ok=True)

        ax1 = fig.axes[0]
        ax1.plot(
            self.pts_voxels[:-1, 1], self.pts_voxels[:-1, 0], linewidth=1, color="white"
        )
        ax1.scatter(self.pts_voxels[0, 1], self.pts_voxels[0, 0], c="white", s=50)
        
        color = "green" if goal_observed else "red"
        ax1.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)
        
        fig.tight_layout()
        plt.savefig(os.path.join(visualization_path, f"{global_step}.png"))
        plt.close()
        
    def save_frontier_visualization(
        self,
        global_step,
        tsdf_planner: TSDFPlanner,
        max_point_choice: Union[SnapShot, Frontier],
        global_caption,
    ):
        assert self.episode_dir is not None
        frontier_video_path = os.path.join(self.episode_dir, "frontier_video")
        episode_frontier_dir = os.path.join(self.episode_dir, "frontier")
        episode_snapshot_dir = os.path.join(self.episode_dir, "snapshot")
        os.makedirs(frontier_video_path, exist_ok=True)
        num_images = len(tsdf_planner.frontiers)
        if type(max_point_choice) == SnapShot:
            num_images += 1
        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if (i < num_images - 1) or (
                    i < num_images and type(max_point_choice) == Frontier
                ):
                    img_path = os.path.join(
                        episode_frontier_dir, tsdf_planner.frontiers[i].image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    if (
                        type(max_point_choice) == Frontier
                        and max_point_choice.image == tsdf_planner.frontiers[i].image
                    ):
                        axs[h_idx, w_idx].set_title("Chosen")
                elif i == num_images - 1 and type(max_point_choice) == SnapShot:
                    img_path = os.path.join(
                        episode_snapshot_dir, max_point_choice.image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    axs[h_idx, w_idx].set_title("Snapshot Chosen")
        fig.suptitle(global_caption, fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(
            os.path.join(frontier_video_path, f"{global_step}.png")
        )
        plt.close()

    def save_result(self, result):
        with open(f'{self.episode_dir}/result.json', 'w') as f:
            json.dump(result, f, indent=2)