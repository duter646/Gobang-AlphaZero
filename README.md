# Gobang AlphaZero V2

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English (English)

### Overview
This project is an implementation of an **AlphaZero-like Reinforcement Learning algorithm** for **Gobang (Five in a Row / Gomoku)**. It uses a combination of **Monte Carlo Tree Search (MCTS)** and a **Residual Neural Network (ResNet)** implemented in PyTorch to output policy probabitilies and evaluate board states through Self-Play. It also includes a visuals using Pygame.

### Features
* **AlphaZero Architecture**: Custom ResNet-based Policy-Value Network + MCTS formulation. **Model Parameters**: 1 initial convolutional layer; 5 residual blocks; policy head consisting of 1 convolutional network layer and 1 fully connected network layer; value head consisting of 1 convolutional network layer and 2 fully connected network layers
* **Heuristic Pruning included**: Optimized specifically for Gobang (immediate win/block detection) to significantly speed up training.
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

> **Note:** The pre-trained PyTorch model `Gobang_AlphaZero_V2_model.pth` will be automatically generated and updated during training. If you already have a model, placing it in the same directory will allow the script to load it automatically.** Currently, the model has been trained for around 1000 iterations, and we will continue to increase the training iterations to further improve the model's performance and release new models in the future.

---

<a name="chinese"></a>
## 中文 (Chinese)

### 简介
本项目是一个基于 **AlphaZero 强化学习算法** 实现的**五子棋 AI** 项目。系统结合了**蒙特卡洛树搜索 (MCTS)** 和基于 PyTorch 实现的**残差神经网络 (ResNet)**，通过自我博弈 (Self-Play) 不断提升模型落子策略和盘面价值评估能力。同时项目内置了基于 Pygame 开发的可视化对战界面。

### 核心特性
* **AlphaZero 架构**：残差块组成的策略价值网络 (Policy-Value Network) 结合 MCTS。
**模型参数**：初始卷积层*1；残差块数*5；策略头1层卷积网络、1层全连接网络；价值头1层卷积网络、2层全连接网络
* **启发式剪枝加速**：加入极速必胜/必防机制启发式搜索，针对五子棋专门优化，极大提升了训练前期的效率和实战胜率。
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

> **提示**：训练过程会自动保存模型权重至当前目录下的 `Gobang_AlphaZero_V2_model.pth`。如果你将已有的训练模型放置在同级目录下，程序在运行时会自动识别并加载以供继续训练或人机实战。**当前训练轮数较少，仅1000轮左右，后续会继续增加训练轮数以提升模型性能，并发布新的模型**