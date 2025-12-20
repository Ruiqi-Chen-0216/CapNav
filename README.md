<div align="center">

# ✨CapNav: Benchmarking Vision Language Models on Capability-conditioned Indoor Navigation✨

<p align="center">
    <a href="https://xiasu.github.io/">Xia Su</a><sup>1*</sup>,
    <a href="https://ruiqi-chen-0216.github.io/">Ruiqi Chen</a><sup>1*</sup>,
    <a href="https://liubl1217.github.io/">Benlin Liu</a><sup>1</sup>,
    <a href="https://jingweim.github.io/">Jingwei Ma</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=5lFDxsMAAAAJ&hl=en">Zonglin Di</a><sup>2</sup>,
    <a href="https://ranjaykrishna.com/index.html">Ranjay Krishna</a><sup>1</sup>,
    <a href="https://jonfroehlich.github.io/">Jon E. Froehlich</a><sup>1</sup>,
    <br>
    <sup>*</sup>Equal Contribution.
    <br>
    <sup>1</sup>University of Washington,
    <sup>2</sup>University of California, Santa Cruz
</p>

<!-- 
<a href='https://arxiv.org/abs/2505.23747'><img src='https://img.shields.io/badge/arXiv-2505.23747-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://diankun-wu.github.io/Spatial-MLLM/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;
-->


![Teaser Visualization](assets/teaser-capnav.png)

</div>
<strong>Capability-Conditioned Navigation (CapNav):</strong> We introduce Capability-Conditioned Navigation (<em><strong>CapNav</strong></em>), a benchmark designed to evaluate how well vision–language models (VLMs) can navigate complex indoor environments given an agent’s specific physical and operational capabilities. As illustrated, CapNav takes as input (1) a tour video of an indoor space, (2) nodes of its navigation graph, (3) an agent’s mobility profile, and (4) a navigation task, and evaluates model outputs along multiple dimensions, including task feasibility, path validity, route traversability, and reasoning validity.
</div>


## 🌟 Overview

![Pipeline Visualization](assets/teaser.png)

</div>

The CapNav benchmark evaluates whether VLMs can correctly ground differences in agent mobility capabilities when generating navigation plans. This example demonstrates a navigation task that has different feasibility and path for different agents.


![Pipeline Visualization](assets/annotation.png)

</div>
Overview of CapNav's data construction: Starting from a 3D indoor scan, we manually record a touring video and a navigation graph. We then use Gemini to generate natural language navigation tasks. Finally, per-task and per-agent traversability are annotated by manually controlling agents in the annotation interface.


## 🎉 Performance

![Results Visualization](assets/performance.png)

## 📦 Dataset

The CapNav benchmark dataset is **not included** in this repository.

All dataset contents are hosted externally:

- [**Hugging Face**](https://huggingface.co/datasets/RichardC0216/CapNav)  
  Structured benchmark data (navigation questions, agent profiles, scene metadata)  

- [**Google Drive**](https://drive.google.com/drive/folders/1NUAE02OPMaf3GnMfXHnuZNktk8cotD4u?usp=sharing)  
  Touring videos of indoor environments  
  (including raw videos and a processed 64-frame @ 1 FPS version for open-source models)

> Note:  
> This repository contains evaluation code and utilities only.  
> Please download the dataset from the links above before running any experiments.


## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/Ruiqi-Chen-0216/CapNav
cd CapNav
```

### 2. Environment Setup

1. **Create conda environment:**

```bash
conda create -n CapNav python=3.10 -y
conda activate CapNav
```

## 🧪 Evaluation

### 1. Data Preparation

Before running any evaluation, you need to prepare the CapNav dataset
and generate capability-conditioned prompts.

#### Download Dataset

The CapNav dataset is hosted on Hugging Face and Google Drive.
Please follow the instructions below to download the structured data.

```bash
huggingface-cli download --resume-download \
  RichardC0216/CapNav \
  --local-dir data/CapNav \
  --repo-type dataset
```
Video data should be downloaded separately from [Google Drive](https://drive.google.com/drive/folders/1NUAE02OPMaf3GnMfXHnuZNktk8cotD4u?usp=sharing).

#### Prompt Generation

CapNav evaluates models using capability-conditioned navigation prompts.
We provide scripts to generate prompts by combining:

- Navigation questions
- Agent profiles
- Scene graph nodes

To generate prompts:

```bash
python scripts/generate_prompts.py 
```

### 2. Evaluation on Open-source Vision–Language Models

We evaluate open-source vision–language models (VLMs) using **preprocessed videos**
sampled to **64 frames at 1 FPS** by default, in order to ensure consistent input length
and fair comparison across models.

All open-source models are evaluated using the same CapNav prompts,
agent profiles, and scene information.
---

#### Running Evaluation

Evaluation is performed via a unified entry script:

```bash
python scripts/run.py --model <MODEL_NAME> --num_frames <NUM_FRAMES> --thinking <on|off>
```

**Arguments:**

- `--model`  
  Name of the open-source vision–language model to evaluate.
- `--num_frames`  
  Number of video frames used as model input.  
  Typical values include `16`, `32`, or `64`, depending on the model’s input capacity
  and available GPU memory.
- `--thinking`  
  Whether to enable internal reasoning mechanisms, if supported by the model.
  Options: `on` or `off`.

**Example:**

```bash
python scripts/run.py --model InternVL3_5-8B --num_frames 32 --thinking on
```
#### Model Loading

Each open-source vision–language model is associated with a corresponding
adapter located at:
```bash
src/model_adapters/<MODEL_NAME>_adapter.py
```

We provide adapters **only for the open-source VLMs evaluated in the paper**.
These adapters are intended to serve as **reference implementations**.
Users who wish to evaluate CapNav on additional open-source models
can extend the benchmark by implementing new adapters following the
existing examples.

By default, pretrained model weights and associated model files are
automatically downloaded from Hugging Face based on the specified `--model`
name when the model is first used.

This behavior is handled by the `ensure_model_downloaded` function
defined in each model adapter, which internally invokes:

```bash
scripts/download_model.sh
```

to manage model downloads.

> **Note:**  
> If you prefer to manually download and deploy model checkpoints locally,
> please comment out or modify the `ensure_model_downloaded` call
> in the corresponding model adapter to disable automatic downloading.



### 3. Evaluation on Peer Spatial Reasoning Models

In addition to vision–language models, CapNav is evaluated on **peer spatial reasoning models**
that explicitly model temporal or spatial reasoning over videos.

These models are evaluated using the same CapNav prompts, agent profiles, and scene information.

Currently evaluated peer models include:
- Spatial-MLLM
- Video-R1

---

#### Spatial-MLLM

We evaluate Spatial-MLLM using its official implementation:

https://github.com/diankun-wu/Spatial-MLLM

All model architectures, checkpoints, and inference logic follow the original repository.
This project does **not** modify the model code.

##### Environment Setup

To simplify deployment, we provide a reference environment setup script:

```bash
source setup_spatialMLLM.sh
```

This script prepares a compatible runtime environment and is provided for convenience only.
Users may need to adjust environment parameters (e.g., CUDA version, GPU architecture)
based on their local setup.

For model-specific configuration and checkpoints, please refer to the original repository [Spatial-MLLM](https://github.com/diankun-wu/Spatial-MLLM).

Running Evaluation

After preparing the environment and CapNav prompts, run:
```bash
python scripts/run_spatial_mllm.py
```

#### Video-R1

We evaluate Video-R1 using its official implementation:

https://github.com/tulerfeng/Video-R1

All model architectures, checkpoints, and inference logic follow the original repository.
This project does **not** modify the model code.

##### Environment Setup

To simplify deployment, we provide a reference environment setup script:

```bash
source setup_videoR1.sh
```

This script prepares a compatible runtime environment and is provided for convenience only.
Users may need to adjust environment parameters (e.g., CUDA version, GPU architecture)
based on their local setup.

For model-specific configuration and checkpoints, please refer to the original repository [Video-R1](https://github.com/tulerfeng/Video-R1).

Running Evaluation

After preparing the environment and CapNav prompts, run:
```bash
python scripts/run_spatial_mllm.py
```


