import os
import sys
import math
import numpy as np
import pygame
import onnxruntime as ort


BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 60
WINDOW_SIZE = (BOARD_SIZE - 1) * CELL_SIZE + 2 * MARGIN

COLOR_BG = (220, 179, 92)
COLOR_GRID = (0, 0, 0)
COLOR_BLACK = (20, 20, 20)
COLOR_WHITE = (240, 240, 240)

DEFAULT_ONNX_NAME = "Gobang_AlphaZero_V2.onnx"
DEFAULT_ONNX_DATA_NAME = "Gobang_AlphaZero_V2.onnx.data"


def choose_font(size):
    pygame.font.init()
    font_candidates = ["microsoftyahei", "simhei", "stxihei", "arialunicodems", "simsun"]
    for name in font_candidates:
        try:
            f = pygame.font.SysFont(name, size)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.SysFont(None, size)


def model_path_for_runtime():
    # 兼容开发环境、PyInstaller onedir(_internal) 以及 exe 同目录放模型
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(exe_dir, DEFAULT_ONNX_NAME),
            os.path.join(exe_dir, "_internal", DEFAULT_ONNX_NAME),
        ]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, DEFAULT_ONNX_NAME),
            os.path.join(script_dir, "_internal", DEFAULT_ONNX_NAME),
        ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # 返回首选路径，便于报错信息清晰
    return candidates[0]


def model_data_path_for_runtime(onnx_path):
    return onnx_path + ".data"


def parse_mode_input(raw, default="1", valid=("1", "2")):
    value = (raw or "").strip()
    if value in valid:
        return value
    return default


def safe_sample_move(acts, probs):
    if len(acts) == 0:
        return None

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.shape[0] != len(acts):
        probs = np.ones(len(acts), dtype=np.float64)

    probs = np.where(np.isfinite(probs), probs, 0.0)
    total = probs.sum()
    if total <= 0:
        probs = np.ones(len(acts), dtype=np.float64) / len(acts)
    else:
        probs = probs / total

    idx = int(np.random.choice(len(acts), p=probs))
    return acts[idx]


class Board:
    def __init__(self):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.last_move = None

    def copy(self):
        b = Board()
        b.state = np.copy(self.state)
        b.current_player = self.current_player
        b.last_move = self.last_move
        return b

    def get_legal_moves(self):
        return list(zip(*np.where(self.state == 0)))

    def execute_move(self, move):
        r, c = move
        self.state[r, c] = self.current_player
        self.last_move = move
        self.current_player = 3 - self.current_player

    def is_winning_move(self, move, player):
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
        win_moves = []
        for move in self.get_legal_moves():
            if self.is_winning_move(move, player):
                win_moves.append(move)
        return win_moves

    def check_win(self):
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
        features = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        features[0] = (self.state == self.current_player).astype(np.float32)
        features[1] = (self.state == (3 - self.current_player)).astype(np.float32)
        if self.last_move:
            features[2][self.last_move[0], self.last_move[1]] = 1.0
        if self.current_player == 1:
            features[3] = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        return features


class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0.0
        self.u = 0.0
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
        self.Q += (leaf_value - self.Q) / self.n_visits
        if self.parent:
            self.parent.backup(-leaf_value)

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=500):
        self.root = TreeNode(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def _playout(self, state):
        node = self.root
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            state.execute_move(move)

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

    def get_move_probs(self, state, temp=1e-3):
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            return (), np.array([])

        immediate_win_moves = state.find_immediate_win_moves(state.current_player)
        if immediate_win_moves:
            forced_move = immediate_win_moves[0]
            return (forced_move,), np.array([1.0], dtype=np.float64)

        opponent = 3 - state.current_player
        opponent_win_moves = state.find_immediate_win_moves(opponent)
        if opponent_win_moves and not immediate_win_moves:
            forced_block_move = opponent_win_moves[0]
            return (forced_block_move,), np.array([1.0], dtype=np.float64)

        for _ in range(self.n_playout):
            state_copy = state.copy()
            self._playout(state_copy)

        if not self.root.children:
            return (), np.array([])

        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)

        if temp == 0:
            probs = np.zeros(len(acts), dtype=np.float64)
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


class ONNXAgent:
    def __init__(self, onnx_path):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"未找到 ONNX 模型文件: {onnx_path}")

        data_path = model_data_path_for_runtime(onnx_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"未找到 ONNX 权重文件: {data_path}\n"
                f"请将 {DEFAULT_ONNX_NAME} 与 {DEFAULT_ONNX_DATA_NAME} 放在同一目录。"
            )

        providers = ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime 初始化失败: {exc}") from exc
        self.input_name = self.session.get_inputs()[0].name

    def policy_value_fn(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return [], 0.0

        features = board.get_features()[np.newaxis, :, :, :].astype(np.float32)
        try:
            policy_log_probs, value = self.session.run(None, {self.input_name: features})
        except Exception as exc:
            raise RuntimeError(f"ONNX 推理失败: {exc}") from exc

        act_probs = np.exp(policy_log_probs[0])
        value = float(value[0][0])

        action_probs = []
        for move in legal_moves:
            idx = move[0] * BOARD_SIZE + move[1]
            action_probs.append((move, float(act_probs[idx])))

        probs = np.array([p for _, p in action_probs], dtype=np.float64)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs = probs / total
            action_probs = [(action_probs[i][0], float(probs[i])) for i in range(len(action_probs))]
        else:
            uniform = 1.0 / len(action_probs)
            action_probs = [(m, uniform) for m, _ in action_probs]

        return action_probs, value


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
            int(c1[2] * (1 - t) + c2[2] * t),
        )

    def _pre_render_stones(self):
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
                if stone == 0:
                    continue
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                radius = CELL_SIZE // 2 - 2

                stone_surf = self.stone_cache[stone]
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


def pve_onnx():
    onnx_path = model_path_for_runtime()
    print(f"正在加载 ONNX 模型: {onnx_path}")

    player_color_raw = input("请选择阵营：1. 执黑(先手) 2. 执白(后手) [默认1]: ")
    player_color = 2 if parse_mode_input(player_color_raw, default="1", valid=("1", "2")) == "2" else 1
    ai_color = 2 if player_color == 1 else 1

    agent = ONNXAgent(onnx_path)
    mcts = MCTS(agent.policy_value_fn, c_puct=5, n_playout=500)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("五子棋 - ONNX 人机对战")
    clock = pygame.time.Clock()
    renderer = GoGameRenderer()

    font = choose_font(48)

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
            current_step = int(np.count_nonzero(board.state))

            if current_step == 0:
                best_move = (BOARD_SIZE // 2, BOARD_SIZE // 2)
            else:
                pygame.event.pump()
                pygame.display.set_caption("五子棋 - AI 思考中...")

                eval_temp = 1e-3 if current_step >= 4 else 0.5
                acts, probs = mcts.get_move_probs(board, temp=eval_temp)
                if len(acts) == 0:
                    winner = -1
                    game_over = True
                    pygame.display.set_caption("五子棋 - 局面结束")
                    continue

                pygame.display.set_caption("五子棋 - 人机对战")

                if eval_temp == 1e-3:
                    best_idx = int(np.argmax(probs))
                    best_move = acts[best_idx]
                else:
                    sampled = safe_sample_move(acts, probs)
                    if sampled is None:
                        winner = -1
                        game_over = True
                        continue
                    best_move = sampled

            board.execute_move(best_move)
            mcts.update_with_move(best_move)
            winner = board.check_win()
            if winner != 0:
                game_over = True

            input_locked = False
            pygame.event.clear(pygame.MOUSEBUTTONDOWN)

        renderer.draw(screen, board.state, board.last_move)
        if game_over:
            draw_game_over_overlay(screen, font, winner, restart_rect, quit_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def pvp():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("五子棋 - 人人对战")
    clock = pygame.time.Clock()
    renderer = GoGameRenderer()

    font = choose_font(48)

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


if __name__ == "__main__":
    try:
        print("=== 五子棋 ===")
        mode = parse_mode_input(input("请选择模式：1. 人机对战 2. 人人对战 [默认1]: "), default="1", valid=("1", "2"))
        if mode == "2":
            pvp()
        else:
            pve_onnx()
    except Exception as exc:
        print(f"运行错误: {exc}")
        input("按回车退出...")
