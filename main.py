# main.py
import os
import random
import asyncio
import argparse
import numpy as np
import shutil
from playwright.async_api import async_playwright
import config
import core
import trainer
from logger_setup import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение агента T-Rex Runner (DQN)")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить обучение с последнего чекпоинта (LOAD_MODEL=True)")
    parser.add_argument("--fresh", action="store_true",
                        help="Начать с нуля, игнорируя чекпоинт (LOAD_MODEL=False). "
                             "Приоритетнее --resume")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Переопределить TOTAL_EPISODES")
    parser.add_argument("--workers", type=int, default=None,
                        help="Переопределить NUM_WORKERS")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Переопределить MAX_STEPS")
    return parser.parse_args()


def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1920, 1080


def cleanup_old_profiles(active_workers):
    upper = max(active_workers + 1, len(config.AGENT_NAMES) + 5)
    for i in range(active_workers, upper):
        folder = os.path.abspath(f"./chrome_profile_{i}")
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass


async def main():
    args = parse_args()

    if args.resume:
        config.LOAD_MODEL = True
    if args.fresh:
        config.LOAD_MODEL = False
    if args.episodes is not None:
        config.TOTAL_EPISODES = args.episodes
    if args.workers is not None:
        config.NUM_WORKERS = args.workers
    if args.max_steps is not None:
        config.MAX_STEPS = args.max_steps

    random.seed(config.SEED)
    np.random.seed(config.SEED)

    cleanup_old_profiles(config.NUM_WORKERS)

    logger.info(
        f"Запускаем {config.NUM_WORKERS} агентов (headless={config.HEADLESS}). "
        f"Эпизодов: {config.TOTAL_EPISODES}"
    )
    core.load_all()

    screen_w, screen_h = get_screen_size()
    center_x = max(0, (screen_w - config.WIN_W) // 2)
    center_y = max(0, (screen_h - config.WIN_H) // 2)

    async with async_playwright() as p:
        contexts = []
        tasks = []
        stop_event = asyncio.Event()

        try:
            for i in range(config.NUM_WORKERS):
                agent_name = config.AGENT_NAMES[i] if i < len(config.AGENT_NAMES) else f"Agent_{i}"
                user_data_dir = os.path.abspath(f"./chrome_profile_{i}")

                is_visible = (i < config.VISIBLE_WORKERS) and not config.HEADLESS
                is_headless = not is_visible

                args_list = [
                    f"--window-size={config.WIN_W},{config.WIN_H}",
                    "--disable-gpu", "--disable-extensions",
                    "--test-type", "--no-sandbox", "--mute-audio",
                    "--disable-session-crashed-bubble", "--hide-crash-restore-bubble",
                    "--no-first-run", "--no-default-browser-check",
                ]

                if is_visible:
                    col = i % 2
                    row = i // 2
                    pos_x = center_x + (col * config.WIN_W) - config.WIN_W // 2
                    pos_y = center_y + (row * config.WIN_H) - config.WIN_H // 2
                    args_list.append(f"--window-position={pos_x},{pos_y}")

                # Без channel="chrome": используем bundled chromium из Playwright.
                context = await p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=is_headless,
                    no_viewport=True,
                    args=args_list,
                )
                contexts.append(context)
                tasks.append(asyncio.create_task(
                    trainer.worker(i + 1, agent_name, context, stop_event)
                ))

                # Пауза, чтобы не перегружать локальный сервер.
                await asyncio.sleep(0.3)

            gather_task = asyncio.gather(*tasks, return_exceptions=True)
            stop_task = asyncio.create_task(stop_event.wait())

            await asyncio.wait([gather_task, stop_task], return_when=asyncio.FIRST_COMPLETED)

            for task in tasks:
                if not task.done():
                    task.cancel()

            # Дожидаемся корректного завершения/отмены всех воркеров.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                    logger.error(f"Воркер {i + 1} завершился с ошибкой: {r}")

        finally:
            for context in contexts:
                try:
                    await context.close()
                except Exception:
                    pass
            if config.TRAIN_MODE:
                await core.save_all()

    logger.info("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")


if __name__ == "__main__":
    asyncio.run(main())
