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

- **Hugging Face**  
  Structured benchmark data (navigation questions, agent profiles, scene metadata)  
  https://huggingface.co/datasets/RichardC0216/CapNav

- **Google Drive**  
  Touring videos of indoor environments  
  (including raw videos and a processed 64-frame @ 1 FPS version for open-source models)

Links to the Google Drive folders are provided in the Hugging Face dataset card.
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


