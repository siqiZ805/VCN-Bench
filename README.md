# VCN-Bench

<div align="center">

<!-- [![HomePage](https://img.shields.io/badge/HomePage-144B9E?logo=ReactOS&logoColor=white)](https://0309hws.github.io/VL-LN.github.io/)
[![Paper](https://img.shields.io/badge/Paper-B31B1B?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2512.22342)
[![Data](https://img.shields.io/badge/Data-FFA500?logo=readthedocs&logoColor=white)](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench/tree/main/)
[![Model](https://img.shields.io/badge/Model-2CA02C?logo=huggingface&logoColor=white)](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel/tree/main) -->
![demo](assets/teaser.png "teaser")

</div>

Spatial reasoning is fundamental to embodied agents, yet it remains unclear whether spatial understanding can be carried forward to guide sequential interactions.
Existing spatial-reasoning benchmarks typically terminate at offline predictions, while navigation benchmarks evaluate spatial reasoning as part of instruction following and exploration.
We introduce VCN-Bench, a <b>V</b>ideo-<b>C</b>ontextualized <b>N</b>avigation benchmark for probing closed-loop spatial reasoning over prior visual experience in MLLMs. Given a prior video covering both the initial location and destination, the agent is tasked with reasoning out the instruction-specified target and navigating toward it with the inferred spatial context. 
Built on Matterport3D, VCN-Bench contains five instruction types, 100k training episodes, and 1,250 evaluation episodes. Navigation serves as the primary evaluation, while diagnostic goal identification helps distinguish destination-resolution errors from subsequent navigation failures. 
We further propose MV-DualVLN, a planning-oriented baseline that jointly leverages prior video and in-episode observations. Experiments reveal limited navigation performance, a substantial destination-resolution-to-navigation gap, and frequent navigation failures even after correct destination identification.

<div align="center">

![demo](assets/bench_demonstration.png "bench")

</div>


## 🛠 Getting Started
We test under the following environment:
* Python 3.9
* Pytorch 2.4.1
* CUDA Version 11.8 

**Preparing  a conda env with `Python3.9` & Install habitat-sim and habitat-lab**
```bash
conda create -n vcn python=3.9
conda install habitat-sim==0.2.5 withbullet headless -c conda-forge -c aihabitat
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines # install habitat_baselines
```

Run the commands below from the repository root. Relative paths in the YAML
configs are resolved from that root, so moving the checkout does not require
editing Python files. Data and local weights default to `datasets/` and
`checkpoints/`; generated runs default to `outputs/`. These directories are
ignored by Git. Set paths in the configs or override the shared defaults:

```bash
export VCN_MP3D_DIR=/path/to/mp3d
export VCN_INSTRUCTIONS=/path/to/instructions.json.gz
export VCN_VIDEO_ROOT=/path/to/tour_rgb
export VCN_OUTPUT_DIR=/path/to/outputs
```

Each run saves its resolved settings in `config.yaml`. All three navigation
entrypoints support `--exp_name`, `--start_ratio`, and `--end_ratio`; the ratios
select a slice of the instruction list. `--split` labels the log file and does
not partition episodes on its own.

## Instruction Dataset Construction

If you want to construct instructions by yourself,
set the scene metadata, viewpoints, tours, trajectories, and output directories
in `config/construct_instructions.yaml`. Run the tools as modules:

```bash
python -m bench_construct.gen_instrs.run --mode train --task obj_instance
python -m bench_construct.gen_instrs.run --mode train --action merge
python -m bench_construct.gen_instrs.run --mode train --action merge --category_only
```



## 📦 Training Data Collection
Set `mp3d_dir`, `file_path`, and `output_parent_dir` in `config/collect_data.yaml`, then collect navigation trajectories:
```bash
python run_collect_data.py --cfg_file config/collect_data.yaml
```

Project the chosen frontier or snapshot at each step to a pixel goal. `frontier_dir` is the episode folder produced above (`output_parent_dir/exp_name`):
```bash
python frontier_to_pixel.py \
  --cfg_file config/collect_data.yaml \
  --frontier_dir outputs/collect_data \
  --instruction datasets/test/instructions.json.gz \
  --output_dir outputs/pixel_labels
```

If several repeats were collected into subfolders of one root, merge them into a single label file:
```bash
python frontier_to_pixel.py --merge --output_root outputs/pixel_labels
```


## 🤖 Evaluation

**MV-DualVLN**

Set `VCN_MVDUALVLN_CHECKPOINT` or `qwen_path` to the VCN-finetuned checkpoint.
This entrypoint uses the custom visual pooling model. The processor defaults
to the original Qwen3-VL-4B processor.
```bash
python run_eval_mvdualvln.py --cfg_file config/eval_mvdualvln.yaml
```

**Zero-shot 3D-Mem**

Set the scene, instruction, video, and detector checkpoints in `config/eval_3dmem.yaml`, then run:
```bash
python run_eval_3dmem.py --cfg_file config/eval_3dmem.yaml
```



## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


## 🙏 Acknowledgement
The navigation pipeline is built upon [3D-Mem](https://github.com/UMass-Embodied-AGI/3D-Mem) (Yang et al., CVPR 2025). We thank the authors for releasing their code.
