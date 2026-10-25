from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import torch
from torch import nn


GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 24
BOARD_WIDTH = GRID_WIDTH * CELL_SIZE
BOARD_HEIGHT = GRID_HEIGHT * CELL_SIZE
PANEL_WIDTH = 340
WINDOW_WIDTH = BOARD_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT
ACTION_NAMES = ("直行", "右转", "左转")
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
MODEL_PATH = Path(__file__).with_name("snake_dqn.pt")


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    score: int


class SnakeEnvironment:
    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> np.ndarray:
        center = (self.width // 2, self.height // 2)
        self.snake = [center, (center[0] - 1, center[1]), (center[0] - 2, center[1])]
        self.direction = 0
        self.score = 0
        self.steps_since_food = 0
        self.visits: dict[tuple[int, int], int] = {}
        self._record_visit(center)
        self.food = self._place_food()
        self.done = False
        return self.get_state()

    def _record_visit(self, position: tuple[int, int]) -> None:
        self.visits[position] = self.visits.get(position, 0) + 1

    def _place_food(self) -> tuple[int, int]:
        occupied = set(self.snake)
        available = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        return random.choice(available) if available else self.snake[0]

    def _turn_direction(self, action: int) -> int:
        if action == 1:
            return (self.direction + 1) % 4
        if action == 2:
            return (self.direction - 1) % 4
        return self.direction

    def _inside(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def _next_position(self, direction: int) -> tuple[int, int]:
        dx, dy = DIRECTIONS[direction]
        return self.snake[0][0] + dx, self.snake[0][1] + dy

    def _reachable_area(self, start: tuple[int, int], body: list[tuple[int, int]]) -> int:
        blocked = set(body[:-1])
        blocked.discard(start)
        frontier = [start]
        visited = {start}
        while frontier:
            x, y = frontier.pop()
            for dx, dy in DIRECTIONS:
                neighbor = (x + dx, y + dy)
                if self._inside(neighbor) and neighbor not in blocked and neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return len(visited)

    def _candidate_is_safe(self, direction: int) -> bool:
        next_position = self._next_position(direction)
        growing = next_position == self.food
        body_to_check = self.snake if growing else self.snake[:-1]
        if not self._inside(next_position) or next_position in body_to_check:
            return False
        simulated = [next_position, *self.snake]
        if not growing:
            simulated.pop()
        reachable = self._reachable_area(next_position, simulated)
        required = min(len(simulated) + 1, self.width * self.height)
        return reachable >= required

    def safe_actions(self) -> list[bool]:
        return [self._candidate_is_safe(self._turn_direction(action)) for action in range(3)]

    def get_state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        forward = self.direction
        right = (forward + 1) % 4
        left = (forward - 1) % 4
        danger = [not self._candidate_is_safe(direction) for direction in (forward, right, left)]
        direction_bits = [int(self.direction == direction) for direction in range(4)]
        food_bits = [
            int(self.food[0] < head_x),
            int(self.food[0] > head_x),
            int(self.food[1] < head_y),
            int(self.food[1] > head_y),
        ]
        distance = [abs(self.food[0] - head_x) / self.width, abs(self.food[1] - head_y) / self.height]
        length_ratio = [len(self.snake) / (self.width * self.height)]
        return np.array([*map(int, danger), *direction_bits, *food_bits, *distance, *length_ratio], dtype=np.float32)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("回合已结束，请先调用 reset()")
        self.direction = self._turn_direction(action)
        next_position = self._next_position(self.direction)
        ate_food = next_position == self.food
        occupied = self.snake if ate_food else self.snake[:-1]
        head_before = self.snake[0]
        old_distance = abs(self.food[0] - head_before[0]) + abs(self.food[1] - head_before[1])
        reward = -0.12
        self.steps_since_food += 1

        if not self._inside(next_position) or next_position in occupied:
            self.done = True
            return StepResult(self.get_state(), -10.0, True, self.score)

        self.snake.insert(0, next_position)
        if ate_food:
            self.score += 1
            reward += 10.0
            self.steps_since_food = 0
            self.food = self._place_food()
        else:
            self.snake.pop()
            new_distance = abs(self.food[0] - next_position[0]) + abs(self.food[1] - next_position[1])
            reward += 0.15 if new_distance < old_distance else -0.15 if new_distance > old_distance else 0.0

        visits = self.visits.get(next_position, 0)
        reward -= min(visits, 5) * 0.08
        self._record_visit(next_position)

        if len(self.snake) == self.width * self.height:
            self.done = True
            reward += 10.0
        elif self.steps_since_food >= self.width * self.height * 2:
            self.done = True
            reward -= 2.0

        return StepResult(self.get_state(), reward, self.done, self.score)


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DQNAgent:
    def __init__(self, state_size: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = QNetwork(state_size, 3).to(self.device)
        self.target = QNetwork(state_size, 3).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=0.001)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.995
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=30_000)
        self.batch_size = 64
        self.train_steps = 0
        self.online.eval()
        if MODEL_PATH.exists():
            self._load()

    def _load(self) -> None:
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            self.online.load_state_dict(checkpoint["model"])
            self.target.load_state_dict(self.online.state_dict())
            self.epsilon = float(checkpoint.get("epsilon", self.epsilon_min))
            print(f"已加载模型：{MODEL_PATH.name}")
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            print(f"模型加载失败，将重新训练：{error}")

    def act(self, state: np.ndarray, safe_actions: list[bool]) -> int:
        valid = [index for index, safe in enumerate(safe_actions) if safe]
        if not valid:
            valid = list(range(3))
        if random.random() < self.epsilon:
            return random.choice(valid)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            values = self.online(state_tensor)[0]
            mask = torch.full_like(values, -torch.inf)
            mask[valid] = values[valid]
            return int(torch.argmax(mask).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.stack([item[0] for item in batch])).to(self.device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([item[3] for item in batch])).to(self.device)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device)

        self.online.train()
        current = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            future = self.target(next_states).max(1).values
            target = rewards + self.gamma * future * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(current, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.online.eval()

        self.train_steps += 1
        if self.train_steps % 250 == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self) -> None:
        torch.save({"model": self.online.state_dict(), "epsilon": self.epsilon}, MODEL_PATH)


class SnakeApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("贪吃蛇 AI | PyTorch DQN")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Microsoft YaHei UI", 18)
        self.small_font = pygame.font.SysFont("Microsoft YaHei UI", 14)
        self.environment = SnakeEnvironment()
        self.agent = DQNAgent(len(self.environment.get_state()))
        self.episode = 0
        self.episode_reward = 0.0
        self.rewards: deque[float] = deque(maxlen=180)
        self.scores: deque[int] = deque(maxlen=100)
        self.last_action = 0
        self.last_loss: float | None = None
        self.running = True

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, BOARD_WIDTH, BOARD_HEIGHT), 1)
        for x in range(0, BOARD_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, (48, 48, 48), (x, 0), (x, BOARD_HEIGHT))
        for y in range(0, BOARD_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, (48, 48, 48), (0, y), (BOARD_WIDTH, y))

        for index, (x, y) in enumerate(self.environment.snake):
            color = (245, 245, 245) if index == 0 else (190, 190, 190)
            pygame.draw.rect(self.screen, color, (x * CELL_SIZE + 3, y * CELL_SIZE + 3, CELL_SIZE - 6, CELL_SIZE - 6))
        food_x, food_y = self.environment.food
        pygame.draw.rect(self.screen, (255, 255, 255), (food_x * CELL_SIZE + 5, food_y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))
        self._draw_panel()
        pygame.display.flip()

    def _draw_panel(self) -> None:
        panel_x = BOARD_WIDTH
        pygame.draw.line(self.screen, (110, 110, 110), (panel_x, 0), (panel_x, WINDOW_HEIGHT), 1)
        x = panel_x + 18
        self._text("训练状态", x, 16, (255, 255, 255), self.font)
        lines = [
            f"回合: {self.episode}",
            f"当前得分: {self.environment.score}",
            f"蛇身长度: {len(self.environment.snake)} / {GRID_WIDTH * GRID_HEIGHT}",
            f"探索率 ε: {self.agent.epsilon:.3f}",
            f"学习率: {self.agent.learning_rate:.4f}",
            f"最近动作: {ACTION_NAMES[self.last_action]}",
            f"最近损失: {self.last_loss:.4f}" if self.last_loss is not None else "最近损失: 收集经验中",
            "", "回合奖励曲线", "白线: 单回合  灰线: 10 回合均值",
        ]
        for index, line in enumerate(lines):
            if line:
                self._text(line, x, 48 + index * 24, (215, 215, 215), self.small_font)
        chart = pygame.Rect(x, 306, PANEL_WIDTH - 36, 240)
        pygame.draw.rect(self.screen, (12, 12, 12), chart)
        pygame.draw.rect(self.screen, (100, 100, 100), chart, 1)
        pygame.draw.line(self.screen, (65, 65, 65), (chart.left, chart.centery), (chart.right, chart.centery))
        if len(self.rewards) > 1:
            values = list(self.rewards)
            limit = max(10.0, max(abs(value) for value in values))
            points = [
                (chart.left + int(index * (chart.width - 2) / (len(values) - 1)) + 1,
                 chart.centery - int(max(-limit, min(limit, value)) / limit * (chart.height // 2 - 8)))
                for index, value in enumerate(values)
            ]
            pygame.draw.lines(self.screen, (240, 240, 240), False, points, 2)
            averages = [sum(values[max(0, index - 9):index + 1]) / len(values[max(0, index - 9):index + 1]) for index in range(len(values))]
            average_points = [
                (chart.left + int(index * (chart.width - 2) / (len(values) - 1)) + 1,
                 chart.centery - int(max(-limit, min(limit, value)) / limit * (chart.height // 2 - 8)))
                for index, value in enumerate(averages)
            ]
            if len(average_points) > 1:
                pygame.draw.lines(self.screen, (115, 115, 115), False, average_points, 2)
        self._text("奖励", chart.left + 6, chart.top + 4, (180, 180, 180), self.small_font)
        self._text("Esc 退出", x, WINDOW_HEIGHT - 30, (160, 160, 160), self.small_font)

    def _text(self, text: str, x: int, y: int, color: tuple[int, int, int], font: pygame.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))

    def _finish_episode(self) -> None:
        self.rewards.append(self.episode_reward)
        self.scores.append(self.environment.score)
        self.episode += 1
        average = sum(list(self.rewards)[-10:]) / min(10, len(self.rewards))
        print(f"回合 {self.episode:04d} | 得分 {self.environment.score:02d} | 奖励 {self.episode_reward:7.2f} | 近10回合均值 {average:7.2f} | ε {self.agent.epsilon:.3f} | 学习率 {self.agent.learning_rate:.4f}")
        self.episode_reward = 0.0
        self.environment.reset()
        if self.episode % 10 == 0:
            self.agent.save()

    def run(self) -> None:
        print(f"设备: {self.agent.device} | 学习率: {self.agent.learning_rate:.4f} | 按 Esc 退出")
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        self.running = False
                if not self.running:
                    break

                state = self.environment.get_state()
                safe_actions = self.environment.safe_actions()
                self.last_action = self.agent.act(state, safe_actions)
                result = self.environment.step(self.last_action)
                self.agent.remember(state, self.last_action, result.reward, result.state, result.done)
                self.last_loss = self.agent.train_step() or self.last_loss
                self.episode_reward += result.reward
                print(f"学习率={self.agent.learning_rate:.4f} | AI按键={ACTION_NAMES[self.last_action]} | 奖励={result.reward:+.2f}", flush=True)
                if result.done:
                    self._finish_episode()
                self.draw()
                self.clock.tick(12)
        finally:
            self.agent.save()
            pygame.quit()


def run_headless_test(episodes: int) -> None:
    environment = SnakeEnvironment()
    steps = 0
    for episode in range(episodes):
        state = environment.reset()
        total_reward = 0.0
        while not environment.done:
            action = random.choice([index for index, safe in enumerate(environment.safe_actions()) if safe] or range(3))
            result = environment.step(action)
            assert result.state.shape == state.shape
            state = result.state
            total_reward += result.reward
            steps += 1
            assert len(environment.snake) == len(set(environment.snake))
            assert all(environment._inside(position) for position in environment.snake)
        print(f"headless 回合 {episode + 1}: 步数={steps}, 得分={environment.score}, 奖励={total_reward:.2f}")
    print(f"无窗口验证通过：{episodes} 回合，累计步数 {steps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch DQN 自动玩贪吃蛇")
    parser.add_argument("--headless-test", action="store_true", help="无窗口运行随机策略环境检查")
    parser.add_argument("--episodes", type=int, default=5, help="无窗口验证回合数")
    args = parser.parse_args()
    if args.headless_test:
        run_headless_test(max(1, args.episodes))
    else:
        SnakeApp().run()


if __name__ == "__main__":
    main()