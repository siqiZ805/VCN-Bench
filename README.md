# TourNav

<div align="center">

[![HomePage](https://img.shields.io/badge/HomePage-144B9E?logo=ReactOS&logoColor=white)](https://0309hws.github.io/VL-LN.github.io/)
[![Paper](https://img.shields.io/badge/Paper-B31B1B?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2512.22342)
[![Data](https://img.shields.io/badge/Data-FFA500?logo=readthedocs&logoColor=white)](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench/tree/main/)
[![Model](https://img.shields.io/badge/Model-2CA02C?logo=huggingface&logoColor=white)](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel/tree/main)
![demo](assets/tournav_teaser.png "demo")

</div>

## 🏠 About


## 📢 News



## 🛠 Getting Started
We test under the following environment:
* Python 3.9
* Pytorch 2.4.1
* CUDA Version 11.8 

1. **Preparing  a conda env with `Python3.9` & Install habitat-sim and habitat-lab**
    ```bash
    conda create -n streamvln python=3.9
    conda install habitat-sim==0.2.5 withbullet headless -c conda-forge -c aihabitat
    git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e habitat-lab  # install habitat_lab
    pip install -e habitat-baselines # install habitat_baselines
    ```

    2. **Clone this repository**


## 📁 Data Preparation

To get started, you need to prepare three types of data:

## 📁 Data Collection

## 🏆 Model Zoo


## 🤖 Evaluation

## 📝 TODO List


## 🔗 Citation


## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements