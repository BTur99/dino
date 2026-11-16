import os
import json
import random
import asyncio
import numpy as np
import tkinter as tk
import shutil
from playwright.async_api import async_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
NUM_WORKERS = 12         # 6 параллельных агентов
TOTAL_EPISODES = 1500    # Увеличили, чтобы агенты успели раскрыть потенциал
ALPHA = 0.15             # Скорость обучения
GAMMA = 0.99             # Дальновидность (теперь агент думает о будущем)
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995    # Дольше исследуем мир (меньше случайностей со временем)

STEP_DELAY = 0.02        # Ускоренная реакция (50 кадров в секунду)
MAX_STEPS = 3000
MODEL_FILE = "q_table_v2.npy"
STATE_FILE = "state_v2.json"

TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True
HEADLESS = False         # Видимые окна
SEED = 42

# --- НАСТРОЙКИ ОКОН ---
WIN_W = 650
WIN_H = 350
OFFSET_X = 350           # Сдвиг вправо от центра

ACTIONS = [0, 1, 2]
NOOP, JUMP, DUCK = 0, 1, 2

q_table = {}
episodes_done = 0
epsilon = EPSILON_START

save_lock = asyncio.Lock()
epsilon_lock = asyncio.Lock()


def get_screen_size():
    try:
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1920, 1080


def cleanup_old_profiles(active_workers):
    for i in range(active_workers, 20):
        folder = os.path.abspath(f"./chrome_profile_{i}")
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass


def get_q_values(state):
    if state not in q_table:
        # 3 действия (NOOP, JUMP, DUCK)
        q_table[state] = [0.0] * len(ACTIONS)
    return q_table[state]


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_all():
    global q_table, episodes_done, epsilon
    if LOAD_MODEL and os.path.exists(MODEL_FILE):
        q_table = np.load(MODEL_FILE, allow_pickle=True).item()
        print(f"Загружена Q-таблица ({len(q_table)} состояний).")
    else:
        print("Q-таблица создана с нуля.")
    if LOAD_MODEL and os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        episodes_done = int(data.get("episodes_done", 0))
        epsilon = float(data.get("epsilon", EPSILON_START))
        print(f"Загружено состояние: эпизодов {episodes_done}, eps {epsilon:.3f}")


async def save_all():
    async with save_lock:
        if not SAVE_MODEL:
            return
        await asyncio.to_thread(np.save, MODEL_FILE, q_table)
        state_data = {"episodes_done": episodes_done, "epsilon": epsilon}
        await asyncio.to_thread(_write_json, STATE_FILE, state_data)
        print(f"[Сохранение] состояний: {len(q_table)} | эпизодов: {episodes_done} | eps: {epsilon:.3f}")


async def decay_epsilon():
    global epsilon
    async with epsilon_lock:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


async def get_state(page):
    try:
        result = await page.evaluate(
            """
            () => {
                const runner = window.Runner?.instance_;
                if (!runner) return { status: "NO_RUNNER" };
                if (runner.crashed) return { status: "DEAD" };
                if (!runner.playing) return { status: "NOT_PLAYING" };

                const dino = runner.tRex;
                const dinoWidth = dino.config ? dino.config.WIDTH : 44;
                const dinoRight = dino.xPos + dinoWidth;
                const obstacles = runner.horizon?.obstacles || [];

                let best = null;
                let bestDist = Infinity;
                for (const o of obstacles) {
                    const d = o.xPos - dinoRight;
                    if (d > -o.width && d < bestDist) {
                        bestDist = d;
                        best = o;
                    }
                }

                if (!best) {
                    return {
                        status: "OK",
                        distance: 1000,
                        speed: runner.currentSpeed,
                        yPos: 100,
                        jumping: dino.jumping,
                        isBird: false,
                        width: 20
                    };
                }

                return {
                    status: "OK",
                    distance: bestDist,
                    speed: runner.currentSpeed,
                    yPos: best.yPos,
                    jumping: dino.jumping,
                    isBird: best.type === "PTERODACTYL" || best.isBird === true,
                    width: best.width
                };
            }
            """
        )

        status = result["status"]
        if status != "OK":
            return None, status, False, status

        distance = result["distance"]
        speed = result["speed"]
        y_pos = result["yPos"]
        jumping = result["jumping"]
        is_bird = result.get("isBird", False)
        width = result.get("width", 20)
        width_bin = 1 if width > 35 else 0
        
        if distance > 220:
            dist_bin = 0
        elif distance > 130:
            dist_bin = 1
        elif distance > 55:
            dist_bin = 2
        else:
            dist_bin = 3

        if speed < 8.5:
            speed_bin = 0
        elif speed < 11.0:
            speed_bin = 1
        else:
            speed_bin = 2

        height_bin = 1 if y_pos < 75 else 0
        bird_bin = 1 if is_bird else 0

        state = (dist_bin, speed_bin, height_bin, bird_bin, width_bin)
        info = f"dist={distance:.0f}, spd={speed:.1f}, y={y_pos}, bird={is_bird}, w={width}"
        return state, info, jumping, "OK"

    except Exception as error:
        return None, f"ERROR: {error}", False, "ERROR"


def choose_action(state, current_epsilon):
    values = get_q_values(state)
    if TRAIN_MODE and random.random() < current_epsilon:
        return random.choice(ACTIONS)
    return int(np.argmax(values))


def update_q_table(state, action, reward, next_state):
    values = get_q_values(state)
    current_q = values[action]
    max_next_q = 0.0 if next_state is None else max(get_q_values(next_state))
    values[action] = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)


async def start_game(page):
    await page.keyboard.up("ArrowDown")
    await asyncio.sleep(0.85)

    info = "UNKNOWN"
    status = "UNKNOWN"
    for _ in range(5):
        await page.keyboard.press("Space")
        await asyncio.sleep(0.15)

        state, info, jumping, status = await get_state(page)
        if state is not None:
            return state, info, jumping, status

    return None, info, False, status


async def worker(worker_id, context, stop_event):
    global episodes_done

    try:
        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda: stop_event.set())
        
        await page.goto(GAME_URL, wait_until="networkidle")
        await asyncio.sleep(1.0)
        
        # Выводим вперед и кликаем для фокуса
        await page.bring_to_front()
        await page.mouse.click(WIN_W // 2, WIN_H // 2)
        await asyncio.sleep(0.5)

        while episodes_done < TOTAL_EPISODES and not stop_event.is_set():
            episodes_done += 1
            ep_num = episodes_done
            current_eps = epsilon if TRAIN_MODE else 0.0

            state, info, jumping, status = await start_game(page)
            
            if stop_event.is_set():
                break

            if state is None:
                print(f"[Агент {worker_id}] Игра не запустилась ({status}). Перезагрузка...")
                await page.reload(wait_until="networkidle")
                await asyncio.sleep(1.0)
                await page.mouse.click(WIN_W // 2, WIN_H // 2)
                continue

            total_reward = 0.0
            last_decision = None

            for step in range(1, MAX_STEPS + 1):
                if stop_event.is_set():
                    break
                    
                if jumping:
                    await asyncio.sleep(STEP_DELAY)
                    next_state, info, jumping, status = await get_state(page)

                    if next_state is None:
                        if status == "DEAD":
                            if TRAIN_MODE and last_decision is not None:
                                prev_state, prev_action = last_decision
                                update_q_table(prev_state, prev_action, -100.0, None)
                            total_reward -= 100.0
                        print(f"[Агент {worker_id} | Игра {ep_num}] Смерть в прыжке ({status}). Шагов: {step} | Награда: {total_reward:.1f}")
                        break

                    state = next_state
                    continue

                action = choose_action(state, current_eps)
                last_decision = (state, action)

                if action == JUMP:
                    await page.keyboard.up("ArrowDown")
                    await page.keyboard.press("Space")
                elif action == DUCK:
                    await page.keyboard.down("ArrowDown")
                else:
                    await page.keyboard.up("ArrowDown")

                await asyncio.sleep(STEP_DELAY)
                next_state, info, jumping, status = await get_state(page)

                # --- УЛУЧШЕНИЕ НАГРАДЫ ---
                if next_state is None:
                    reward = -100.0 if status == "DEAD" else 0.0
                else:
                    # Если просто бежит — 1.0, если прыгает/приседает — 0.9 (легкий штраф за лишние движения)
                    reward = 1.0 if action == 0 else 0.9 

                if TRAIN_MODE:
                    update_q_table(state, action, reward, next_state)

                total_reward += reward
                state = next_state

                if next_state is None:
                    print(f"[Агент {worker_id} | Игра {ep_num}] Конец ({status}). Шагов: {step} | Награда: {total_reward:.1f}")
                    break

            if not stop_event.is_set():
                try:
                    await page.keyboard.up("ArrowDown")
                except Exception:
                    pass

            if TRAIN_MODE and not stop_event.is_set():
                await decay_epsilon()
                # Сохраняем реже (каждые 100 эпизодов)
                if ep_num % 20 == 0:
                    await save_all()

    except Exception as error:
        if not stop_event.is_set():
            print(f"[Агент {worker_id}] Окно закрыто или произошла ошибка: {error}")
        stop_event.set()


async def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    cleanup_old_profiles(NUM_WORKERS)
    
    print(f"Запускаем {NUM_WORKERS} окон друг на друге...")
    load_all()

    screen_w, screen_h = get_screen_size()
    
    center_x = max(0, (screen_w - WIN_W) // 2)
    pos_x = center_x + OFFSET_X
    pos_x = min(pos_x, screen_w - WIN_W)
    pos_y = max(0, (screen_h - WIN_H) // 2)

    async with async_playwright() as p:
        contexts = []
        tasks = []
        stop_event = asyncio.Event()

        try:
            for i in range(NUM_WORKERS):
                x = pos_x
                y = pos_y
                
                user_data_dir = os.path.abspath(f"./chrome_profile_{i}")

                context = await p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    channel="chrome",
                    headless=HEADLESS,
                    no_viewport=True,
                    args=[
                        f"--window-size={WIN_W},{WIN_H}",
                        f"--window-position={x},{y}",
                        "--disable-gpu",
                        "--disable-extensions",
                        "--test-type",
                        "--no-sandbox",
                        "--mute-audio", # Отключаем звук, чтобы 6 окон не создавали кашу
                    ],
                )
                contexts.append(context)
                tasks.append(asyncio.create_task(worker(i + 1, context, stop_event)))

            gather_task = asyncio.gather(*tasks, return_exceptions=True)
            stop_task = asyncio.create_task(stop_event.wait())
            
            await asyncio.wait(
                [gather_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in tasks:
                if not task.done():
                    task.cancel()

        finally:
            for context in contexts:
                try:
                    await context.close()
                except Exception:
                    pass
            if TRAIN_MODE:
                await save_all()

    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО (или прервано пользователем)!")


if __name__ == "__main__":
    asyncio.run(main())