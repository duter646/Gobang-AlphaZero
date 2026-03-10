import pygame
import numpy as np
import sys
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque

# --- 配置 ---
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 60
WINDOW_SIZE = (BOARD_SIZE - 1) * CELL_SIZE + 2 * MARGIN

COLOR_BG = (220, 179, 92)
COLOR_GRID = (0, 0, 0)
COLOR_BLACK = (20, 20, 20)
COLOR_WHITE = (240, 240, 240)

FILEPATH_MODEL = "Gobang_AlphaZero_V2_model.pth"


# ==========================================
# 1. 游戏逻辑与状态表示
# ==========================================
class Board:
    def __init__(self):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1 # 1: 黑, 2: 白
        self.last_move = None

    def copy(self):
        new_board = Board()
        new_board.state = np.copy(self.state)
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        return new_board

    def get_legal_moves(self):
        return list(zip(*np.where(self.state == 0)))

    def execute_move(self, move):
        r, c = move
        self.state[r, c] = self.current_player
        self.last_move = move
        self.current_player = 3 - self.current_player # 切换玩家 1->2, 2->1

    def is_winning_move(self, move, player):
        """判断 player 落在 move 后是否立即形成五连。"""
        r, c = move
        if self.state[r, c] != 0:
            return False

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1

            nr, nc = r + dr, c + dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.state[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc

            nr, nc = r - dr, c - dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.state[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 5:
                return True

        return False

    def find_immediate_win_moves(self, player):
        """返回 player 当前所有一步成五的落点。"""
        winning_moves = []
        for move in self.get_legal_moves():
            if self.is_winning_move(move, player):
                winning_moves.append(move)
        return winning_moves

    def check_win(self):
        """返回: 1(黑胜), 2(白胜), -1(平局), 0(未结束)"""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.state[r, c] == 0:
                    continue
                color = self.state[r, c]
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                for dr, dc in directions:
                    count = 1
                    for i in range(1, 5):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.state[nr, nc] == color:
                            count += 1
                        else:
                            break
                    if count >= 5:
                        return color
        if len(self.get_legal_moves()) == 0:
            return -1
        return 0

    def get_features(self):
        """
        返回当前视角的特征图 (4, 15, 15)
        """
        features = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        features[0] = (self.state == self.current_player).astype(np.float32)
        features[1] = (self.state == (3 - self.current_player)).astype(np.float32)
        if self.last_move:
            features[2][self.last_move[0], self.last_move[1]] = 1.0
        if self.current_player == 1:
            features[3] = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        return features

# ==========================================
# 2. 策略价值网络 (Policy-Value Network)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        
        # 增加初始通道数 64 -> 128 (利用剩余显存)
        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # 增加残差块数量 3 -> 5 (提升模型容量和表达能力)
        self.res_blocks = nn.Sequential(
            *[ResBlock(128) for _ in range(5)]
        )
        
        # 策略头
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1) # 通道数 2 -> 4
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 128) # 隐藏层 64 -> 128
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy = F.log_softmax(p, dim=1)
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value

# ==========================================
# 3. 蒙特卡洛树搜索 (MCTS) - 加入必胜剪枝启发式
# ==========================================
class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0  
        self.Q = 0         
        self.u = 0         
        self.P = prior_p   

    def expand(self, action_priors):
        for move, prob in action_priors:
            if move not in self.children:
                self.children[move] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        self.u = c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u

    def backup(self, leaf_value):
        self.n_visits += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits
        if self.parent:
            self.parent.backup(-leaf_value)

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400): # 增加默认 playout 400 -> 500
        self.root = TreeNode(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def _playout(self, state):
        node = self.root
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            state.execute_move(move)

        # 正常使用神经网络评估
        action_probs, leaf_value = self.policy_value_fn(state)
        
        end_status = state.check_win()
        if end_status == 0:
            node.expand(action_probs)
        else:
            if end_status == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if end_status == state.current_player else -1.0

        node.backup(-leaf_value)

    def get_move_probs(self, state, temp=1e-3, add_noise=False):
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            return (), np.array([])

        # 极速必胜剪枝：当前玩家若有一步成五，直接强制该点 100%
        immediate_win_moves = state.find_immediate_win_moves(state.current_player)
        if immediate_win_moves:
            forced_move = immediate_win_moves[0]
            return (forced_move,), np.array([1.0], dtype=np.float64)

        # 极速必防剪枝：若对手存在一步成五点，直接强制优先堵点
        opponent = 3 - state.current_player
        opponent_win_moves = state.find_immediate_win_moves(opponent)
        if opponent_win_moves and not immediate_win_moves:
            forced_block_move = opponent_win_moves[0]
            return (forced_block_move,), np.array([1.0], dtype=np.float64)

        if add_noise:
            if self.root.is_leaf():
                action_probs, _ = self.policy_value_fn(state)
                self.root.expand(action_probs)
            noise = np.random.dirichlet([0.03] * len(self.root.children))
            for i, (move, node) in enumerate(self.root.children.items()):
                node.P = 0.75 * node.P + 0.25 * noise[i]

        for n in range(self.n_playout):
            state_copy = state.copy()
            self._playout(state_copy)

        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        
        if temp == 0:
            best_act = acts[np.argmax(visits)]
            probs = np.zeros(len(acts))
            probs[np.argmax(visits)] = 1.0
            return acts, probs
            
        visits = np.array(visits, dtype=np.float64)
        log_visits = np.log(visits + 1e-10)
        log_probs = (1.0 / temp) * log_visits
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        return acts, probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

# ==========================================
# 4. 训练与自我博弈引擎
# ==========================================
class AlphaZeroAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AlphaZero V2 Using device: {self.device}")
        
        self.net = PolicyValueNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-4) # 稍微降低 L2 正则，鼓励拟合大网络
        
        if os.path.exists(FILEPATH_MODEL):
            self.net.load_state_dict(torch.load(FILEPATH_MODEL, map_location=self.device))
            print(f"Loaded existing V2 model: {FILEPATH_MODEL}")
        else:
            print("Starting fresh training for V2 architecture.")

    def policy_value_fn(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return [], 0.0
            
        features = board.get_features()
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_act_probs, value = self.net(features_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            value = value.item()
            
        action_probs = []
        for move in legal_moves:
            idx = move[0] * BOARD_SIZE + move[1]
            action_probs.append((move, act_probs[idx]))
            
        return action_probs, value

    def train_step(self, state_batch, mcts_probs_batch, winner_batch):
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_batch)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        log_act_probs, value = self.net(state_batch)
        
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, dim=1))
        
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()

    def save_model(self):
        torch.save(self.net.state_dict(), FILEPATH_MODEL)

def self_play_episode(agent, n_playout=600, render=False, renderer=None, screen=None, clock=None, font=None, ep=0, episodes=0):
    board = Board()
    mcts = MCTS(agent.policy_value_fn, c_puct=5, n_playout=n_playout)
    
    states, mcts_probs, current_players = [], [], []
    step = 0 
    
    while True:
        if render:
            # 高频处理事件，防止可视化训练时窗口假死
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        temp = 1.0

        # 强制 AI 执黑第一步下天元
        if step == 0:
            move = (BOARD_SIZE // 2, BOARD_SIZE // 2)
            acts = board.get_legal_moves()
            probs = np.zeros(len(acts))
            probs[acts.index(move)] = 1.0
        else:
            if step < 15:
                temp = 1.0
            elif step < 30:
                temp = 0.5
            else:
                temp = 1e-3

            acts, probs = mcts.get_move_probs(board, temp=temp)

        if render:
            # 缓解 MCTS 搜索过程中的长时间阻塞导致的未响应
            pygame.event.pump()
        
        states.append(board.get_features())
        
        prob_array = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for act, p in zip(acts, probs):
            prob_array[act[0] * BOARD_SIZE + act[1]] = p
        mcts_probs.append(prob_array)
        
        current_players.append(board.current_player)
        
        # 每步只采样一次，并保持棋盘与 MCTS 同步更新
        act_idx = np.random.choice(len(acts), p=probs)
        move = acts[act_idx]

        board.execute_move(move)
        mcts.update_with_move(move)
        step += 1

        if render:
            renderer.draw(screen, board.state, board.last_move)
            info_text = f"Ep: {ep}/{episodes} | Step: {step} | Temp: {temp}"
            text_surf = font.render(info_text, True, (0, 0, 0))
            screen.blit(text_surf, (10, 10))
            pygame.display.flip()
            clock.tick(10) 
        
        winner = board.check_win()
        if winner != 0:
            if render:
                draw_game_over_overlay(screen, font, winner)
                pygame.display.flip()
                time.sleep(1)
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            return states, mcts_probs, winners_z

def train_alphazero(episodes=2000, batch_size=1024, render=False): # 增大 Batch Size 以更好利用大网络和长对局历史
    agent = AlphaZeroAgent()
    data_buffer = deque(maxlen=20000) # 将经验池扩大到 20000 充分利用显存
    
    renderer, screen, clock, font = None, None, None, None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("AlphaZero V2 可视化训练")
        clock = pygame.time.Clock()
        pygame.font.init()
        font_list = ["microsoftyahei", "simhei", "stxihei", "arialunicodems"]
        for name in font_list:
            try:
                font = pygame.font.SysFont(name, 32)
                if font: break
            except:
                continue
        if not font:
            font = pygame.font.SysFont(None, 32)
        renderer = GoGameRenderer()
    
    for ep in range(episodes):
        print(f"Episode {ep+1}/{episodes} - Self Playing...")
        n_playout = 500
        if render:
            pygame.event.pump()
            n_playout = 400
        states, mcts_probs, winners_z = self_play_episode(
            agent, n_playout=n_playout, 
            render=render, renderer=renderer, screen=screen, clock=clock, font=font,
            ep=ep+1, episodes=episodes
        )
        
        for i in range(len(states)):
            s, p, w = states[i], mcts_probs[i], winners_z[i]
            p_2d = p.reshape(BOARD_SIZE, BOARD_SIZE)
            for j in range(4):
                s_rot = np.array([np.rot90(c, j) for c in s])
                p_rot = np.rot90(p_2d, j).flatten()
                data_buffer.append((s_rot, p_rot, w))
                s_flip = np.array([np.fliplr(c) for c in s_rot])
                p_flip = np.fliplr(p_rot.reshape(BOARD_SIZE, BOARD_SIZE)).flatten()
                data_buffer.append((s_flip, p_flip, w))
                
        print(f"Episode {ep+1} finished. Buffer size: {len(data_buffer)}. Winner: {'黑' if winners_z[0] == 1 else '白' if winners_z[0] == -1 else '平局'}")
        
        if len(data_buffer) > batch_size:
            mini_batch = random.sample(data_buffer, batch_size)
            state_batch, mcts_probs_batch, winner_batch = zip(*mini_batch)
            loss, p_loss, v_loss = agent.train_step(state_batch, mcts_probs_batch, winner_batch)
            print(f"Loss: {loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f})\n")
            
        if (ep + 1) % 10 == 0:
            agent.save_model()
            print("Model saved.\n")

# ==========================================
# 5. Pygame 渲染与人机对战
# ==========================================
class GoGameRenderer:
    def __init__(self):
        self.board_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE)).convert()
        self._pre_render_board()
        self.stone_cache = {}
        self._pre_render_stones()

    def _mix_color(self, c1, c2, t):
        t = max(0.0, min(1.0, t))
        return (
            int(c1[0] * (1 - t) + c2[0] * t),
            int(c1[1] * (1 - t) + c2[1] * t),
            int(c1[2] * (1 - t) + c2[2] * t)
        )

    def _pre_render_stones(self):
        # 放大渲染后缩小，利用平滑缩放消除锯齿
        radius = CELL_SIZE // 2 - 2
        scale_factor = 4
        big_radius = radius * scale_factor
        surf_size = big_radius * 2 + 2

        import pygame.gfxdraw

        for stone in [1, 2]:
            big_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            
            base = COLOR_BLACK if stone == 1 else COLOR_WHITE
            ambient = self._mix_color(COLOR_BG, COLOR_WHITE, 0.18)
            warm_ambient = self._mix_color(COLOR_BG, COLOR_WHITE, 0.28)

            if stone == 1:
                center_color = self._mix_color(base, ambient, 0.27)
                edge_color = self._mix_color(base, COLOR_BLACK, 0.22)
                outline_color = self._mix_color(base, ambient, 0.13)
            else:
                center_color = self._mix_color(base, warm_ambient, 0.10)
                edge_color = self._mix_color(base, COLOR_BLACK, 0.12)
                outline_color = self._mix_color(base, COLOR_BLACK, 0.18)

            center = (surf_size // 2, surf_size // 2)

            for r in range(big_radius, 1, -1):
                t = r / big_radius
                t = t ** 1.28
                color = self._mix_color(center_color, edge_color, t)
                pygame.draw.circle(big_surface, color, center, r)

            pygame.gfxdraw.aacircle(big_surface, center[0], center[1], big_radius, outline_color)
            
            color_at_edge = self._mix_color(center_color, edge_color, 1.0)
            pygame.gfxdraw.aacircle(big_surface, center[0], center[1], big_radius - 1, color_at_edge)

            # 平滑缩小到正常尺寸
            final_surface = pygame.transform.smoothscale(big_surface, (radius * 2 + 1, radius * 2 + 1))
            self.stone_cache[stone] = final_surface
    def _pre_render_board(self):
        self.board_surface.fill(COLOR_BG)
        for i in range(BOARD_SIZE):
            start = MARGIN + i * CELL_SIZE
            end = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE
            pygame.draw.line(self.board_surface, COLOR_GRID, (start, MARGIN), (start, end), 1)
            pygame.draw.line(self.board_surface, COLOR_GRID, (MARGIN, start), (end, start), 1)

    def draw(self, screen, board_state, last_move=None):
        screen.blit(self.board_surface, (0, 0))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = board_state[r, c]
                if stone == 0: continue
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                radius = CELL_SIZE // 2 - 2
                
                # 直接块贴图，快且边缘超滑
                stone_surf = self.stone_cache[stone]
                # 调整居中位置
                screen.blit(stone_surf, (x - radius, y - radius))
                
                if last_move and last_move == (r, c):
                    import pygame.gfxdraw
                    pygame.gfxdraw.aacircle(screen, x, y, 4, (255, 0, 0))
                    pygame.gfxdraw.filled_circle(screen, x, y, 4, (255, 0, 0))

def draw_game_over_overlay(screen, font, winner, restart_rect=None, quit_rect=None):
    overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    screen.blit(overlay, (0, 0))

    if winner == 1:
        title = "黑方获胜！"
    elif winner == 2:
        title = "白方获胜！"
    else:
        title = "平局！"
        
    title_surf = font.render(title, True, (255, 255, 255))
    title_rect = title_surf.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 60))
    screen.blit(title_surf, title_rect)

    if restart_rect and quit_rect:
        pygame.draw.rect(screen, (30, 30, 30), restart_rect)
        pygame.draw.rect(screen, (30, 30, 30), quit_rect)
        pygame.draw.rect(screen, (220, 220, 220), restart_rect, 2)
        pygame.draw.rect(screen, (220, 220, 220), quit_rect, 2)

        btn_font = pygame.font.SysFont("microsoftyahei", 32)
        restart_text = btn_font.render("继续游戏", True, (255, 255, 255))
        quit_text = btn_font.render("退出游戏", True, (255, 255, 255))
        screen.blit(restart_text, restart_text.get_rect(center=restart_rect.center))
        screen.blit(quit_text, quit_text.get_rect(center=quit_rect.center))

def pve():
    player_color = input("请选择您的阵营：1. 执黑(先手) 2. 执白(后手) [默认1]: ")
    player_color = 2 if player_color.strip() == '2' else 1
    ai_color = 2 if player_color == 1 else 1
    
    print("正在加载 AlphaZero V2 模型...")
    agent = AlphaZeroAgent()
    mcts = MCTS(agent.policy_value_fn, c_puct=5, n_playout=500) # 实战 PVE 全力以赴，跑 500 次模拟

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("五子棋 - AlphaZero V2 人机对战")
    clock = pygame.time.Clock()
    renderer = GoGameRenderer()
    
    pygame.font.init()
    font = pygame.font.SysFont("microsoftyahei", 48)

    board = Board()
    running = True
    game_over = False
    winner = 0

    button_w, button_h = 160, 50
    restart_rect = pygame.Rect(WINDOW_SIZE // 2 - button_w - 10, WINDOW_SIZE // 2 + 10, button_w, button_h)
    quit_rect = pygame.Rect(WINDOW_SIZE // 2 + 10, WINDOW_SIZE // 2 + 10, button_w, button_h)
    input_locked = False

    while running:
        is_ai_turn = (board.current_player == ai_color) and not game_over
        if is_ai_turn:
            input_locked = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if game_over:
                    if restart_rect.collidepoint(mx, my):
                        board = Board()
                        mcts = MCTS(agent.policy_value_fn, c_puct=5, n_playout=500)
                        game_over = False
                        winner = 0
                    elif quit_rect.collidepoint(mx, my):
                        running = False
                    continue

                if (not is_ai_turn) and (not input_locked):
                    c = round((mx - MARGIN) / CELL_SIZE)
                    r = round((my - MARGIN) / CELL_SIZE)
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        move = (r, c)
                        if move in board.get_legal_moves():
                            board.execute_move(move)
                            mcts.update_with_move(move)
                            winner = board.check_win()
                            if winner != 0:
                                game_over = True

        if is_ai_turn and not game_over:
            current_step = sum(row.count(1) + row.count(2) for row in board.state.tolist())
            
            # PVE 时，强制 AI 执黑第一步下天元
            if current_step == 0:
                best_move = (BOARD_SIZE // 2, BOARD_SIZE // 2)
            else:
                # PVE 时，为了防止界面卡死未响应，每次可以只算一部分，用多线程的话太复杂，这里简单每算一段就抛一抛事件
                # 在单线程简单办法就是把 playout 拆开或者减少。如果是实战卡顿，通常是因为 2000 次深度导致计算过久
                # 为了在 Pygame 下防止卡死，可以在思考前设置界面状态并在思考中抛出事件，但最直接的是降低 playout 或简化思考循环
                # 我们将 pve 里的思考过程加上屏幕更新，或者你可以接受在后台算，我们只需在这里调用 pygame.event.pump()
                pygame.event.pump()
                pygame.display.set_caption("五子棋 - AI 正在疯狂算力中... (请稍等)")
                
                eval_temp = 1e-3 if current_step >= 4 else 0.5
                
                mcts.n_playout = 500
                acts, probs = mcts.get_move_probs(board, temp=eval_temp)
                
                pygame.display.set_caption("五子棋 - AlphaZero V2 人机对战")
                
                if eval_temp == 1e-3:
                    best_idx = np.argmax(probs)
                else:
                    best_idx = np.random.choice(len(acts), p=probs)
                    
                best_move = acts[best_idx]
            
            board.execute_move(best_move)
            mcts.update_with_move(best_move)
            winner = board.check_win()
            if winner != 0:
                game_over = True

            # AI 落子后解锁输入，并清空思考期间积压的点击事件
            input_locked = False
            pygame.event.clear(pygame.MOUSEBUTTONDOWN)

        renderer.draw(screen, board.state, board.last_move)
        if game_over:
            draw_game_over_overlay(screen, font, winner, restart_rect, quit_rect)
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

def pvp():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("五子棋 - 人人对战")
    clock = pygame.time.Clock()
    renderer = GoGameRenderer()
    
    pygame.font.init()
    font_list = ["microsoftyahei", "simhei", "stxihei", "arialunicodems"]
    font = None
    for name in font_list:
        try:
            font = pygame.font.SysFont(name, 32)
            if font: break
        except:
            continue
    if not font:
        font = pygame.font.SysFont(None, 32)

    board = Board()
    running = True
    game_over = False
    winner = 0

    button_w, button_h = 160, 50
    restart_rect = pygame.Rect(WINDOW_SIZE // 2 - button_w - 10, WINDOW_SIZE // 2 + 10, button_w, button_h)
    quit_rect = pygame.Rect(WINDOW_SIZE // 2 + 10, WINDOW_SIZE // 2 + 10, button_w, button_h)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if game_over:
                    if restart_rect.collidepoint(mx, my):
                        board = Board()
                        game_over = False
                        winner = 0
                    elif quit_rect.collidepoint(mx, my):
                        running = False
                    continue

                c = round((mx - MARGIN) / CELL_SIZE)
                r = round((my - MARGIN) / CELL_SIZE)
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    move = (r, c)
                    if move in board.get_legal_moves():
                        board.execute_move(move)
                        winner = board.check_win()
                        if winner != 0:
                            game_over = True

        renderer.draw(screen, board.state, board.last_move)
        if game_over:
            draw_game_over_overlay(screen, font, winner, restart_rect, quit_rect)
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    print("=== 五子棋 AlphaZero (MCTS) V2 ===")
    mode = int(input("请选择模式：1.人机对战 2.人人对战 3.训练AI 4.可视化训练 (输入数字，输入其他退出)："))
    if mode == 1:
        pve()
    elif mode == 2:
        pvp()
    elif mode == 3:
        train_alphazero(episodes=int(input("请输入训练回合数：")), render=False)
    elif mode == 4:
        train_alphazero(episodes=int(input("请输入可视化训练回合数：")), render=True)
    else:
        print("无效输入，程序退出。")

