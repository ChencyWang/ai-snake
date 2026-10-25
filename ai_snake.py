from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import torch
from torch import nn
from torch.nn import functional as functional

from snake_game import (
    COLORS,
    GRID_HEIGHT,
    GRID_WIDTH,
    PAUSE_BUTTON,
    WINDOW_SIZE,
    choose_food,
    draw_game,
    draw_text,
)


MODEL_PATH = Path(__file__).with_name("snake_dqn.pt")
STATE_SIZE = 11
ACTION_COUNT = 3
BATCH_SIZE = 64
REPLAY_CAPACITY = 20_000
GAMMA = 0.92
LEARNING_RATE = 0.001
STEP_PENALTY = 0.01
DISTANCE_REWARD = 0.25
FOOD_REWARD = 10.0
COLLISION_PENALTY = -10.0
ACTION_KEYS = ("↑", "→", "←")

# Actions are relative to the current direction: straight, turn right, turn left.
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class SnakeEnvironment:
    """Small headless Snake environment used for DQN self-play."""

    def __init__(self, max_steps: int = 800) -> None:
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> np.ndarray:
        center_x = GRID_WIDTH // 2
        center_y = GRID_HEIGHT // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction_index = 1
        self.food = choose_food(self.snake)
        self.score = 0
        self.steps = 0
        self.done = False
        return self.state()

    def board(self) -> np.ndarray:
        occupied = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
        for x, y in self.snake:
            occupied[y, x] = True
        return occupied

    def next_head(self, action: int) -> tuple[int, int]:
        if action == 1:
            direction_index = (self.direction_index + 1) % len(DIRECTIONS)
        elif action == 2:
            direction_index = (self.direction_index - 1) % len(DIRECTIONS)
        else:
            direction_index = self.direction_index
        dx, dy = DIRECTIONS[direction_index]
        return self.snake[0][0] + dx, self.snake[0][1] + dy

    def collides(self, position: tuple[int, int]) -> bool:
        x, y = position
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return True
        body = self.snake if position == self.food else self.snake[:-1]
        return position in body

    def safe_actions(self) -> list[int]:
        return [action for action in range(ACTION_COUNT) if not self.collides(self.next_head(action))]

    def state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        danger = [
            float(self.collides(self.next_head(action)))
            for action in (0, 1, 2)
        ]
        direction_flags = [float(self.direction_index == index) for index in range(4)]
        food_flags = [0.0] * 4 if self.food is None else [
            float(self.food[0] < head_x),
            float(self.food[0] > head_x),
            float(self.food[1] < head_y),
            float(self.food[1] > head_y),
        ]
        return np.asarray(danger + direction_flags + food_flags, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, int]:
        if self.done:
            raise RuntimeError("Reset the environment before stepping a finished game.")

        if action == 1:
            self.direction_index = (self.direction_index + 1) % len(DIRECTIONS)
        elif action == 2:
            self.direction_index = (self.direction_index - 1) % len(DIRECTIONS)

        old_distance = abs(self.food[0] - self.snake[0][0]) + abs(self.food[1] - self.snake[0][1])
        dx, dy = DIRECTIONS[self.direction_index]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        if self.collides(new_head):
            self.done = True
            self.steps += 1
            return self.state(), COLLISION_PENALTY, True, self.score

        self.snake.insert(0, new_head)
        reward = -STEP_PENALTY
        if new_head == self.food:
            self.score += 1
            reward += FOOD_REWARD
            self.food = choose_food(self.snake)
            if self.food is None:
                self.done = True
        else:
            self.snake.pop()
            new_distance = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])
            distance_change = old_distance - new_distance
            if distance_change > 0:
                reward += DISTANCE_REWARD
            elif distance_change < 0:
                reward -= DISTANCE_REWARD

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
        return self.state(), reward, self.done, self.score


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DQNAgent:
    def __init__(self) -> None:
        self.network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=REPLAY_CAPACITY)
        self.epsilon = 1.0
        self.gamma = GAMMA
        self.updates = 0

    def choose_action(self, state: np.ndarray, safe_actions: list[int], explore: bool) -> int:
        candidates = safe_actions or list(range(ACTION_COUNT))
        if explore and random.random() < self.epsilon:
            return random.choice(candidates)

        state_tensor = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            values = self.network(state_tensor).squeeze(0).cpu().numpy()
        masked_values = np.full(ACTION_COUNT, -np.inf, dtype=np.float32)
        masked_values[candidates] = values[candidates]
        return int(np.argmax(masked_values))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> None:
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states = torch.from_numpy(np.stack([item[0] for item in batch]))
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([item[3] for item in batch]))
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

        current_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_values * (1.0 - dones)

        loss = functional.smooth_l1_loss(current_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % 250 == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def save(self, path: Path | None = None) -> None:
        torch.save(self.network.state_dict(), path or MODEL_PATH)

    def load(self, path: Path | None = None) -> None:
        self.network.load_state_dict(torch.load(path or MODEL_PATH, map_location="cpu", weights_only=True))
        self.target_network.load_state_dict(self.network.state_dict())
        self.network.eval()


class TrainingDashboard:
    WIDTH = 900
    HEIGHT = 560

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("贪吃蛇 AI | 实时学习曲线")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.active = True

    def close(self) -> None:
        if self.active:
            self.active = False
            pygame.display.quit()

    def update(
        self,
        episode: int,
        total_episodes: int,
        scores: list[int],
        recent_average: float,
        best_score: int,
        epsilon: float,
        learning_rate: float,
        last_key: str,
    ) -> None:
        if not self.active:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return

        self.screen.fill((17, 31, 32))
        draw_text(self.screen, "DQN TRAINING", 24, COLORS["green"], (32, 22), bold=True)
        draw_text(
            self.screen,
            f"训练 {episode}/{total_episodes}    最近 50 局平均 {recent_average:.2f}    最高分 {best_score}",
            16,
            COLORS["text"],
            (34, 68),
        )
        draw_text(
            self.screen,
            f"学习率 {learning_rate:.5f}    探索率 ε {epsilon:.3f}    最近方向键 {last_key}",
            14,
            COLORS["muted"],
            (34, 98),
        )

        chart = pygame.Rect(72, 158, self.WIDTH - 112, self.HEIGHT - 220)
        pygame.draw.rect(self.screen, (22, 40, 39), chart, border_radius=5)
        visible_scores = scores[-500:]
        maximum_score = max(5, ((max(visible_scores, default=0) + 4) // 5) * 5)
        for grid_index in range(6):
            y = chart.bottom - round(chart.height * grid_index / 5)
            pygame.draw.line(self.screen, (48, 68, 62), (chart.left, y), (chart.right, y), 1)
            draw_text(self.screen, str(round(maximum_score * grid_index / 5)), 11, COLORS["muted"], (chart.left - 34, y - 8))

        if len(visible_scores) > 1:
            score_points = []
            average_points = []
            for index, score in enumerate(visible_scores):
                x = chart.left + round(chart.width * index / (len(visible_scores) - 1))
                score_y = chart.bottom - round(chart.height * score / maximum_score)
                start = max(0, len(scores) - len(visible_scores) + index - 49)
                average = sum(scores[start : len(scores) - len(visible_scores) + index + 1]) / min(50, len(scores) - len(visible_scores) + index + 1)
                average_y = chart.bottom - round(chart.height * average / maximum_score)
                score_points.append((x, score_y))
                average_points.append((x, average_y))
            pygame.draw.lines(self.screen, (103, 166, 91), False, score_points, 2)
            pygame.draw.lines(self.screen, COLORS["green"], False, average_points, 3)

        draw_text(self.screen, "单局得分", 12, (103, 166, 91), (chart.left + 8, chart.top + 8))
        draw_text(self.screen, "最近 50 局平均", 12, COLORS["green"], (chart.left + 90, chart.top + 8))
        draw_text(self.screen, "Esc 关闭曲线窗口，训练继续在命令行运行", 12, COLORS["muted"], (32, self.HEIGHT - 34))
        pygame.display.flip()


def action_key(environment: SnakeEnvironment, action: int) -> str:
    if action == 1:
        direction_index = (environment.direction_index + 1) % len(DIRECTIONS)
    elif action == 2:
        direction_index = (environment.direction_index - 1) % len(DIRECTIONS)
    else:
        direction_index = environment.direction_index
    return ("↑", "→", "↓", "←")[direction_index]


def train_agent(episodes: int, max_steps: int) -> DQNAgent:
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    agent = DQNAgent()
    environment = SnakeEnvironment(max_steps=max_steps)
    best_score = 0
    recent_scores: deque[int] = deque(maxlen=50)
    scores: list[int] = []
    total_steps = 0
    dashboard = TrainingDashboard()
    learning_rate = agent.optimizer.param_groups[0]["lr"]

    print(f"开始自我对弈训练：{episodes} 局。训练结束后模型保存到 {MODEL_PATH.name}")
    for episode in range(1, episodes + 1):
        state = environment.reset()
        score = 0
        episode_reward = 0.0
        last_key = "-"
        while not environment.done:
            safe_actions = environment.safe_actions()
            action = agent.choose_action(state, safe_actions, explore=True)
            last_key = action_key(environment, action)
            next_state, reward, done, score = environment.step(action)
            episode_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1
            if total_steps % 4 == 0:
                agent.learn()
            agent.epsilon = max(0.05, agent.epsilon * 0.9995)

        recent_scores.append(score)
        scores.append(score)
        best_score = max(best_score, score)
        average_score = sum(recent_scores) / len(recent_scores)
        dashboard.update(episode, episodes, scores, average_score, best_score, agent.epsilon, learning_rate, last_key)
        print(
            f"训练 {episode:4}/{episodes} 局 | 得分 {score:3} | 总奖励 {episode_reward:7.2f} | "
            f"最近平均 {average_score:5.2f} | 学习率 {learning_rate:.5f} | "
            f"探索率 ε {agent.epsilon:.3f} | AI按键 {last_key}",
            flush=True,
        )

    agent.save()
    dashboard.close()
    return agent


def reset_environment(environment: SnakeEnvironment) -> np.ndarray:
    return environment.reset()


def play_game(agent: DQNAgent) -> None:
    pygame.init()
    pygame.display.set_caption("贪吃蛇 AI | PyTorch DQN")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    environment = SnakeEnvironment(max_steps=10_000)
    state = environment.state()
    paused = False
    accumulator = 0.0
    step_seconds = 0.1
    running = True

    while running:
        elapsed = min(clock.tick(60) / 1000, 0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if PAUSE_BUTTON.collidepoint(event.pos) and not environment.done:
                    paused = not paused
                    accumulator = 0.0
                    print("鼠标按键: 左键暂停" if paused else "鼠标按键: 左键继续", flush=True)
            elif event.type == pygame.KEYDOWN:
                print(f"键盘按键: {pygame.key.name(event.key)}", flush=True)
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p and not environment.done:
                    paused = not paused
                    accumulator = 0.0
                elif environment.done and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    state = reset_environment(environment)
                    paused = False
                    accumulator = 0.0

        if not paused and not environment.done:
            accumulator += elapsed
            while accumulator >= step_seconds and not environment.done:
                accumulator -= step_seconds
                action = agent.choose_action(state, environment.safe_actions(), explore=False)
                print(
                    f"AI按键: {action_key(environment, action)} | "
                    f"学习率: {agent.optimizer.param_groups[0]['lr']:.5f}",
                    flush=True,
                )
                state, _reward, _done, _score = environment.step(action)

        draw_game(
            screen,
            environment.snake,
            environment.food,
            environment.score,
            environment.score,
            paused,
            environment.done,
        )
        pygame.draw.rect(screen, COLORS["background"], (0, WINDOW_SIZE[1] - 30, WINDOW_SIZE[0], 30))
        draw_text(screen, "AI 自动游玩     P 暂停     Esc 退出", 13, COLORS["muted"], (16, WINDOW_SIZE[1] - 24))
        draw_text(screen, "AI", 14, COLORS["green"], (105, 22), bold=True)
        pygame.display.flip()

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 NumPy 和 PyTorch DQN 自我对弈并游玩贪吃蛇")
    parser.add_argument("--episodes", type=int, default=500, help="自我对弈训练局数，默认 500")
    parser.add_argument("--max-steps", type=int, default=800, help="每局训练最多移动步数，默认 800")
    parser.add_argument("--train", action="store_true", help="即使已有模型也重新训练")
    parser.add_argument("--train-only", action="store_true", help="训练并保存模型后退出，不打开游戏窗口")
    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes 和 --max-steps 必须大于 0")

    if args.train or not MODEL_PATH.exists():
        agent = train_agent(args.episodes, args.max_steps)
    else:
        agent = DQNAgent()
        agent.load()
        print(f"已加载模型：{MODEL_PATH.name}")

    if not args.train_only:
        play_game(agent)


if __name__ == "__main__":
    main()
