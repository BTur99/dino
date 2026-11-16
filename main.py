import os
import random
import asyncio
import numpy as np
import tkinter as tk
import shutil
from playwright.async_api import async_playwright
import config
import core
import trainer

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

async def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    
    cleanup_old_profiles(config.NUM_WORKERS)
    
    print(f"Запускаем {config.NUM_WORKERS} окон друг на друге...")
    core.load_all()

    screen_w, screen_h = get_screen_size()
    
    center_x = max(0, (screen_w - config.WIN_W) // 2)
    pos_x = center_x + config.OFFSET_X
    pos_x = min(pos_x, screen_w - config.WIN_W)
    pos_y = max(0, (screen_h - config.WIN_H) // 2)

    async with async_playwright() as p:
        contexts = []
        tasks = []
        stop_event = asyncio.Event()

        try:
            for i in range(config.NUM_WORKERS):
                agent_name = config.AGENT_NAMES[i]
                user_data_dir = os.path.abspath(f"./chrome_profile_{i}")

                context = await p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    channel="chrome",
                    headless=config.HEADLESS,
                    no_viewport=True,
                    args=[
                        f"--window-size={config.WIN_W},{config.WIN_H}",
                        f"--window-position={pos_x},{pos_y}",
                        "--disable-gpu",
                        "--disable-extensions",
                        "--test-type",
                        "--no-sandbox",
                        "--mute-audio",
                    ],
                )
                contexts.append(context)
                tasks.append(asyncio.create_task(trainer.worker(i + 1, agent_name, context, stop_event)))

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
            if config.TRAIN_MODE:
                await core.save_all()

    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО (или прервано пользователем)!")

if __name__ == "__main__":
    asyncio.run(main())