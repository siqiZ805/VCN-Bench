"""Common configuration, command-line, and logging setup."""

import argparse
import logging
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = {
    "output_parent_dir",
    "mp3d_dir",
    "file_path",
    "test_data_path",
    "video_root",
    "concept_graph_config_path",
    "scene_dataset_config_path",
    "yolo_model_name",
    "sam_model_name",
    "clip_model_path",
    "scene_info_path",
    "viewpoints_dir",
    "regions_path",
    "tour_video_root",
    "trajectory_dir",
    "region_dist_path",
    "output_dir",
    "sim_config_path",
    "agent_config_path",
}


def project_path(path):
    """Resolve a resource path independently of the working directory."""
    path = Path(os.path.expandvars(str(path))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_config_paths(cfg):
    for key in PATH_FIELDS:
        if cfg.get(key):
            cfg[key] = str(project_path(cfg[key]))
    # Hugging Face model IDs remain IDs; local checkpoint directories become absolute.
    for key in ("qwen_path", "qwen_processor_path"):
        if cfg.get(key):
            checkpoint = str(cfg[key])
            if (
                checkpoint.startswith((".", "~", "checkpoints/"))
                or Path(checkpoint).is_absolute()
                or project_path(checkpoint).exists()
            ):
                cfg[key] = str(project_path(checkpoint))
    return cfg


def load_config(path):
    from omegaconf import OmegaConf

    path = Path(path).expanduser()
    if not path.exists():
        path = project_path(path)
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return resolve_config_paths(cfg)


def configure_environment():
    for key, value in {
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "HABITAT_SIM_LOG": "quiet",
        "MAGNUM_LOG": "quiet",
    }.items():
        os.environ.setdefault(key, value)


def select_episode_range(episodes, start_ratio=0.0, end_ratio=1.0):
    if not 0.0 <= start_ratio < end_ratio <= 1.0:
        raise ValueError(
            "Episode ratios must satisfy 0 <= start_ratio < end_ratio <= 1"
        )
    return episodes[int(len(episodes) * start_ratio) : int(len(episodes) * end_ratio)]


class ElapsedTimeFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s - %(message)s")
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        hours, remainder = divmod(max(0, int(record.created - self.start_time)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def run_cli(main, default_config):
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "-cf", "--cfg_file", default=default_config, help="YAML configuration"
    )
    parser.add_argument("--start_ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", default=1.0, type=float)
    parser.add_argument(
        "--split", default=1, type=int, help="Shard label used in the log filename"
    )
    parser.add_argument(
        "--exp_name", default=None, help="Override the experiment directory name"
    )
    args = parser.parse_args()
    select_episode_range([], args.start_ratio, args.end_ratio)
    cfg = load_config(args.cfg_file)
    if args.exp_name:
        cfg.exp_name = args.exp_name
    cfg.output_dir = str(Path(cfg.output_parent_dir) / cfg.exp_name)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the resolved config, including CLI overrides, without invoking a shell.
    OmegaConf.save(cfg, output_dir / "config.yaml")
    from src.io_utils import save_json

    save_json(vars(args), output_dir / "run_arguments.json")
    log_path = (
        output_dir / f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log"
    )
    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    formatter = ElapsedTimeFormatter()
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logging.info("Running %s", cfg.exp_name)
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=args.split)
