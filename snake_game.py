from __future__ import annotations

import random

import pygame


CELL_SIZE = 24
GRID_WIDTH = 25
GRID_HEIGHT = 20
PLAY_WIDTH = CELL_SIZE * GRID_WIDTH
PLAY_HEIGHT = CELL_SIZE * GRID_HEIGHT
HEADER_HEIGHT = 64
WINDOW_SIZE = (PLAY_WIDTH, PLAY_HEIGHT + HEADER_HEIGHT)

COLORS = {
    "background": (17, 31, 32),
    "board": (22, 40, 39),
    "grid": (34, 55, 52),
    "panel": (27, 47, 45),
    "text": (239, 245, 224),
    "muted": (151, 174, 157),
    "green": (167, 224, 113),
    "green_dark": (103, 166, 91),
    "food": (255, 121, 100),
    "overlay": (12, 24, 25),
}

PAUSE_BUTTON = pygame.Rect(PLAY_WIDTH - 96, 13, 80, 38)


def choose_food(snake: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Return a free grid cell, or None if the board is full."""
    occupied = set(snake)
    available = [
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
        if (x, y) not in occupied
    ]
    return random.choice(available) if available else None


def draw_text(
    surface: pygame.Surface,
    text: str,
    size: int,
    color: tuple[int, int, int],
    position: tuple[int, int],
    *,
    centered: bool = False,
    bold: bool = False,
) -> None:
    font = pygame.font.SysFont("microsoftyahei", size, bold=bold)
    image = font.render(text, True, color)
    rect = image.get_rect()
    if centered:
        rect.center = position
    else:
        rect.topleft = position
    surface.blit(image, rect)


def draw_game(
    screen: pygame.Surface,
    snake: list[tuple[int, int]],
    food: tuple[int, int] | None,
    score: int,
    best_score: int,
    paused: bool,
    game_over: bool,
) -> None:
    screen.fill(COLORS["background"])
    pygame.draw.rect(screen, COLORS["panel"], (0, 0, PLAY_WIDTH, HEADER_HEIGHT))
    draw_text(screen, "贪吃蛇", 21, COLORS["green"], (18, 17), bold=True)
    draw_text(screen, f"得分 {score:03}", 17, COLORS["text"], (185, 21), bold=True)
    draw_text(screen, f"最高 {best_score:03}", 14, COLORS["muted"], (305, 23))
    button_color = COLORS["green_dark"] if paused else (48, 75, 65)
    pygame.draw.rect(screen, button_color, PAUSE_BUTTON, border_radius=6)
    draw_text(screen, "继续" if paused else "暂停", 15, COLORS["text"], PAUSE_BUTTON.center, centered=True, bold=True)

    board_rect = pygame.Rect(0, HEADER_HEIGHT, PLAY_WIDTH, PLAY_HEIGHT)
    pygame.draw.rect(screen, COLORS["board"], board_rect)
    for x in range(0, PLAY_WIDTH + 1, CELL_SIZE):
        pygame.draw.line(screen, COLORS["grid"], (x, HEADER_HEIGHT), (x, HEADER_HEIGHT + PLAY_HEIGHT))
    for y in range(0, PLAY_HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(screen, COLORS["grid"], (0, HEADER_HEIGHT + y), (PLAY_WIDTH, HEADER_HEIGHT + y))

    if food is not None:
        food_rect = pygame.Rect(
            food[0] * CELL_SIZE + 5,
            HEADER_HEIGHT + food[1] * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10,
        )
        pygame.draw.ellipse(screen, COLORS["food"], food_rect)
        pygame.draw.circle(screen, (255, 205, 145), (food_rect.centerx - 3, food_rect.centery - 3), 2)

    for index, (x, y) in enumerate(reversed(snake)):
        inset = 2 if index else 1
        rect = pygame.Rect(
            x * CELL_SIZE + inset,
            HEADER_HEIGHT + y * CELL_SIZE + inset,
            CELL_SIZE - inset * 2,
            CELL_SIZE - inset * 2,
        )
        color = COLORS["green"] if index == len(snake) - 1 else COLORS["green_dark"]
        pygame.draw.rect(screen, color, rect, border_radius=6)

    if paused or game_over:
        veil = pygame.Surface((PLAY_WIDTH, PLAY_HEIGHT), pygame.SRCALPHA)
        veil.fill((*COLORS["overlay"], 205))
        screen.blit(veil, (0, HEADER_HEIGHT))
        title = "游戏结束" if game_over else "暂停中"
        hint = "按 Enter 重新开始   Esc 退出" if game_over else "按 P 或点击按钮继续"
        draw_text(screen, title, 38, COLORS["text"], (PLAY_WIDTH // 2, HEADER_HEIGHT + 155), centered=True, bold=True)
        draw_text(screen, hint, 16, COLORS["muted"], (PLAY_WIDTH // 2, HEADER_HEIGHT + 205), centered=True)

    draw_text(screen, "方向键 / WASD 移动     P 暂停     Esc 退出", 13, COLORS["muted"], (16, WINDOW_SIZE[1] - 24))
    pygame.display.flip()


def run_game(speed: int, best_score: int) -> int:
    pygame.init()
    pygame.display.set_caption("贪吃蛇 | SNAKE")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2), (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2), (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)]
    direction = (1, 0)
    next_direction = direction
    food = choose_food(snake)
    score = 0
    paused = False
    game_over = False
    accumulator = 0.0
    step_seconds = 1 / speed
    running = True

    key_directions = {
        pygame.K_UP: (0, -1),
        pygame.K_w: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_s: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_a: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_d: (1, 0),
    }

    while running:
        elapsed = min(clock.tick(60) / 1000, 0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if PAUSE_BUTTON.collidepoint(event.pos) and not game_over:
                    paused = not paused
                    accumulator = 0.0
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif game_over and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2), (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2), (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)]
                    direction = (1, 0)
                    next_direction = direction
                    food = choose_food(snake)
                    score = 0
                    paused = False
                    game_over = False
                    accumulator = 0.0
                elif event.key == pygame.K_p and not game_over:
                    paused = not paused
                    accumulator = 0.0
                elif not game_over and event.key in key_directions:
                    requested = key_directions[event.key]
                    if requested != (-direction[0], -direction[1]):
                        next_direction = requested

        if not paused and not game_over:
            accumulator += elapsed
            while accumulator >= step_seconds:
                accumulator -= step_seconds
                direction = next_direction
                head = snake[0]
                new_head = (head[0] + direction[0], head[1] + direction[1])
                eating = new_head == food
                body_to_check = snake if eating else snake[:-1]
                if (
                    not 0 <= new_head[0] < GRID_WIDTH
                    or not 0 <= new_head[1] < GRID_HEIGHT
                    or new_head in body_to_check
                ):
                    game_over = True
                    break
                snake.insert(0, new_head)
                if eating:
                    score += 1
                    food = choose_food(snake)
                    if food is None:
                        game_over = True
                else:
                    snake.pop()

        draw_game(screen, snake, food, score, max(best_score, score), paused, game_over)

    pygame.quit()
    return score


def main() -> None:
    run_game(speed=10, best_score=0)


if __name__ == "__main__":
    main()

