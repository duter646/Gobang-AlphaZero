# Gobang AlphaZero V2

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English (English)

### Overview
This project is an implementation of an **AlphaZero-like Reinforcement Learning algorithm** for **Gobang (Five in a Row / Gomoku)**. It uses a combination of **Monte Carlo Tree Search (MCTS)** and a **Residual Neural Network (ResNet)** implemented in PyTorch to output policy probabitilies and evaluate board states through Self-Play. It also includes a beautifully rendered physical-style UI and anti-aliased visuals using Pygame.

### Features
* **AlphaZero Architecture**: Custom ResNet-based Policy-Value Network + MCTS formulation.
* **Heuristic Pruning included**: Optimized specifically for Gobang (immediate win/block detection) to significantly speed up training.
* **Graphical Interface**: High-quality visual effects (shadow and light on stones) using Pygame.
* **Multiple Modes**:
    1. Player vs AI (PvE)
    2. Player vs Player (PvP)
    3. Headless Training (Self-play without rendering)
    4. Visualized Training (Self-play with rendering)

### Installation
1. Clone this repository.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Run the main script in your terminal:
```bash
python Gobang_AlphaZero_V2.py
```
Follow the prompt in the terminal to select the desired mode (1-4).

### Play with ONNX (Recommended for Players)
For users who just want to play without installing heavy dependencies like PyTorch, **or simply download the Release package**:
1. **Source Version**: Install minimal dependencies first:
   ```bash
   pip install numpy>=1.19.0
   pip install pygame>=2.0.0
   pip install onnxruntime>=1.18.0
   ```
   Then run `python Gobang_AlphaZero_V2_ONNX_Play.py`. Make sure `Gobang_AlphaZero_V2.onnx` and `Gobang_AlphaZero_V2.onnx.data` are in the same folder.
2. **EXE Version**: Download the compressed package from the **Releases** page. Extract it and run `Gobang_AlphaZero_V2_ONNX_Play.exe`. This version is standalone and does not require Python.

> **Note:** The pre-trained PyTorch model `Gobang_AlphaZero_V2_model.pth` will be automatically generated and updated during training. If you already have a model, placing it in the same directory will allow the script to load it automatically. **The current provided model is trained for ~4000 episodes; newer versions will be released later.**

---

<a name="chinese"></a>
## 中文 (Chinese)

### 简介
本项目是一个基于 **AlphaZero 强化学习算法** 实现的**五子棋 AI** 项目。系统结合了**蒙特卡洛树搜索 (MCTS)** 和基于 PyTorch 实现的**残差神经网络 (ResNet)**，通过自我博弈 (Self-Play) 不断提升模型落子策略和盘面价值评估能力。同时项目内置了基于 Pygame 开发的精美物理质感与抗锯齿处理的可视化对战界面。

### 核心特性
* **AlphaZero 架构**：残差块组成的策略价值网络 (Policy-Value Network) 结合 MCTS。
* **启发式剪枝加速**：加入极速必胜/必防机制启发式搜索，针对五子棋专门优化，极大提升了训练前期的效率和实战胜率。
* **高质量界面**：使用 Pygame 绘制，棋子具有丰富的光影厚度质感与抗锯齿处理，对战体验极佳。
* **多功能模式**：
    1. 人机对战 (PvE)
    2. 双人对战 (PvP)
    3. 后台无头训练模式 (高效 Self-play)
    4. 带有界面的可视化训练模式

### 安装环境
1. 克隆本项目代码。
2. 安装所需的依赖库（建议使用虚拟环境）：
   ```bash
   pip install -r requirements.txt
   ```

### 运行指南
在终端中运行主程序：
```bash
python Gobang_AlphaZero_V2.py
```
根据终端交互提示输入数字，选择对应的模式进行游玩或开始训练。

### 游玩说明 (ONNX 版本推荐)
如果你仅是为了游玩而非训练，建议使用 ONNX 版本的脚本，**或直接下载release包**，可以免去安装庞大的 PyTorch 依赖。
1. **源码运行**：先按照上方说明安装环境，或不安装PyTorch（安装环境时输入下方命令）
   ```bash
   pip install numpy>=1.19.0
   pip install pygame>=2.0.0
   pip install onnxruntime>=1.18.0
   ```
   直接运行 `python Gobang_AlphaZero_V2_ONNX_Play.py`。请确保 `Gobang_AlphaZero_V2.onnx` 和 `Gobang_AlphaZero_V2.onnx.data` 文件位于同级目录下。
2. **EXE 运行**：在 GitHub 的 **Releases** 页面下载打包好的压缩包。解压后直接双击 `Gobang_AlphaZero_V2_ONNX_Play.exe` 即可运行，无需安装 Python 环境。

> **提示**：训练过程会自动保存模型权重至当前目录下的 `Gobang_AlphaZero_V2_model.pth`。如果你将已有的训练模型放置在同级目录下，程序在运行时会自动识别并加载以供继续训练或人机实战。**当前版本为训练约4000轮的模型，后续会发布新版本**
