# GeoReasoning: Generalizable Geometric Image Caption Synthesis
<link rel="stylesheet" href="./static/css/bulma.min.css">
<link rel="stylesheet" href="./static/css/bulma-carousel.min.css">
<link rel="stylesheet" href="./static/css/bulma-slider.min.css">
<link rel="stylesheet" href="./static/css/fontawesome.all.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
<link rel="stylesheet" href="./static/css/index.css">

<p align="center">
  📑 <a href="https://machinephoenix.github.io/GeoReasoning_blog/">Blog</a> &nbsp&nbsp
  📑 <a href="https://arxiv.org/abs/2509.15217">Paper</a> &nbsp&nbsp
  🤗 <a href="https://huggingface.co/datasets/ScaleMath/GeoReasoning">Hugging Face</a>
</p>

# Introduction

* 📊 GeoReasoning delivers high-quality image-caption pairs that outperform all counterparts on downstream benchmarks with superior scaling.
* 🌐 Achieves significant gains beyond geometry - boosting performance in non-geometric math tasks (2.8-4.8%) and non-mathematical domains like art & engineering (2.4-3.9%).
* ⚡ Built from 50 basic relations, enabling unlimited complexity expansion for diverse geometry problems.


<div align="center">
  <img src="figs/mmmu_bar.jpg" width="90%" alt="mmmu" />
</div>
<div align="center">
  <img src="figs/mathvista_scaling.jpg" width="45%" alt="mathvista" />
  <img src="figs/mathverse_scaling.jpg" width="45%" alt="mathverse" />
</div>


This repository contains the official implementation for the GeoReasoning dataset and training framework, which significantly enhances multimodal reasoning capabilities in AI systems, particularly for geometric problem solving.

# Data Generation
The data generation pipeline is shown below:
<div align="center">
  <img src="figs/generation.jpg" width="80%" alt="generation" />
</div>

Some generated samples are exhibited here:
<div align="center">
  <img src="figs/ex1.png" width="28%" alt="ex1" />
  <img src="figs/ex2.png" width="30%" alt="ex2" />
  <img src="figs/ex3.png" width="25%" alt="ex3" />
</div>

The code will be released very soon.

# RLVR
The training pipeline is:
<div align="center">
  <img src="figs/raft.jpg" width="80%" alt="raft" />
</div>
where the reward modeling is shown below:
<div align="center">
  <img src="figs/reward.jpg" width="80%" alt="reward" />
</div>


Our implementation is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a very strong codebase for fine-tuning and RL.


## Installation
We adopt the installation of LLaMA-Factory.

> [!IMPORTANT]
> Installation is mandatory.

<details><summary>Install from Source</summary>

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

Extra dependencies available: torch, torch-npu, metrics, deepspeed, liger-kernel, bitsandbytes, hqq, eetq, gptq, aqlm, vllm, sglang, galore, apollo, badam, adam-mini, qwen, minicpm_v, openmind, swanlab, dev
</details>

<details><summary>Install from Docker Image</summary>
```bash
docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
```

This image is built on Ubuntu 22.04 (x86\_64), CUDA 12.4, Python 3.11, PyTorch 2.6.0, and Flash-attn 2.7.4.

Find the pre-built images: https://hub.docker.com/r/hiyouga/llamafactory/tags

Please refer to [build docker](#build-docker) to build the image yourself.
</details>



<details><summary>Setting up a virtual environment with <b>uv</b></summary>

Create an isolated Python environment with [uv](https://github.com/astral-sh/uv):

```bash
uv sync --extra torch --extra metrics --prerelease=allow
```

Run LLaMA-Factory in the isolated environment:

```bash
uv run --prerelease=allow llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml
```

</details>


## QuickStart
Gemma3-Infer contains all necessary codes for SFT and VLLM inference.

```
└── Gemma3-Infer
    ├── scripts_raft
        ├── run_coldstart.sh
    │   └── run_raft.sh
    ├── src_raft
        ├── caption_generation_llamafactory_ray.py
        ├── reasoning_rewarding_ray_noStatistics_bystep_updateBest.py
        ├── add_data_item.py
        ├── data0_processing.py
        └── caption_reward.py
```

The scripts_raft folder contains shell files for coldstart (run_coldstart.sh) and RLVR (run_raft.sh), where the latter controls the workflow of RLVR and splits the process into three sub-stages, i.e., caption generation, rewarding, and re-training. 

The src_raft folder contains python files, where add_data_item.py and data0_processing.py is the pre-processing called in run_raft.sh. caption_generation_llamafactory_ray.py and reasoning_rewarding_ray_noStatistics_bystep_updateBest.py implement the caption generation and the reward modeling stage of RLVR with the use of VLLM and ray, respectively. caption_reward.py computes the caption reward during the reward modeling process.


## Citation
```
@misc{georeasoning,
      title={Generalizable Geometric Image Caption Synthesis}, 
      author={Yue Xin and Wenyuan Wang and Rui Pan and Ruida Wang and Howard Meng and Shizhe Diao and Renjie Pi and Tong Zhang},
      year={2025},
      eprint={2509.15217},
      archivePrefix={arXiv},
      primaryClass={cs.AI; cs.CV; cs.LG},
      url={https://arxiv.org/abs/2509.15217}, 
  }
```

## Acknowledgement

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [VLLM](https://github.com/vllm-project/vllm), [AlphaGeometry](https://github.com/google-deepmind/alphageometry), and [AutoGeo](https://github.com/AutoGeo-Official/AutoGeo). Thanks for their wonderful works.

