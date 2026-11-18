# trainer.py
import asyncio
import numpy as np
import config
import core
import game_env
from logger_setup import logger

async def worker(worker_id, agent_name, context, stop_event):
    try:
        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda: stop_event.set())
        await page.add_init_script(f"document.title = '{agent_name}';")
        
        # Используем domcontentloaded для локального сервера
        try:
            await page.goto(config.GAME_URL, wait_until="domcontentloaded", timeout=10000)
        except:
            await asyncio.sleep(2)
            await page.goto(config.GAME_URL, wait_until="domcontentloaded", timeout=15000)
            
        await asyncio.sleep(1.0)
        
        # Если окно видимое — рисуем имя
        if worker_id <= config.VISIBLE_WORKERS and not config.HEADLESS:
            await page.evaluate(f"""
                () => {{
                    const div = document.createElement('div');
                    div.innerText = '{agent_name}';
                    div.style.position = 'absolute'; div.style.top = '100px'; div.style.left = '50%';
                    div.style.transform = 'translateX(-50%)'; div.style.fontSize = '40px';
                    div.style.fontWeight = 'bold'; div.style.color = 'rgba(0, 0, 0, 0.15)';
                    div.style.pointerEvents = 'none'; div.style.zIndex = '9999';
                    document.body.appendChild(div);
                }}
            """)
            await page.bring_to_front()
            await page.mouse.click(config.WIN_W // 2, config.WIN_H // 2)
        
        await asyncio.sleep(0.5)

        while core.episodes_done < config.TOTAL_EPISODES and not stop_event.is_set():
            core.episodes_done += 1
            ep_num = core.episodes_done
            current_eps = core.epsilon if config.TRAIN_MODE else 0.0

            state_tuple, info, jumping, ttc, speed, status = await game_env.start_game(page)
            if stop_event.is_set(): break
            if state_tuple is None: continue
            
            state = np.array(state_tuple, dtype=np.float32)
            total_reward = 0.0
            step = 0

            for step in range(1, config.MAX_STEPS + 1):
                if stop_event.is_set(): break
                    
                action = core.choose_action(state, current_eps)
                current_ttc = ttc
                current_speed = speed 
                current_bird_bin = state[2] 

                if action == config.JUMP:
                    if not jumping: await page.keyboard.press("Space")
                elif action == config.DUCK:
                    await page.keyboard.down("ArrowDown")
                else:
                    await page.keyboard.up("ArrowDown")

                await asyncio.sleep(config.STEP_DELAY)
                next_state_tuple, info, jumping, ttc, speed, status = await game_env.get_state(page)

                done = False
                if next_state_tuple is None:
                    if status == "DEAD":
                        reward = config.REWARD_DEATH
                        done = True
                    else:
                        reward = 0.0
                else:
                    next_state = np.array(next_state_tuple, dtype=np.float32)
                    
                    # --- ЖЕСТКАЯ СИСТЕМА НАГРАД ---
                    reward = config.REWARD_ALIVE + (current_speed * config.REWARD_SPEED_BONUS)
                    
                    # Проверяем, где он был ДО действия. jumping берется из прошлого шага.
                    if jumping:
                        if action == config.DUCK:
                            reward += config.REWARD_FAST_FALL
                        else:
                            reward -= 0.5 
                    else:
                        if action == config.JUMP:
                            if current_ttc > 40:
                                reward += config.REWARD_EARLY_JUMP
                            elif current_ttc < 15:
                                reward += config.REWARD_LATE_JUMP
                            else:
                                reward += config.REWARD_GOOD_JUMP
                        elif action == config.DUCK:
                            if current_bird_bin == 0:
                                reward += config.REWARD_USELESS_DUCK

                if config.TRAIN_MODE:
                    next_state_arr = np.zeros(5, dtype=np.float32) if done else next_state
                    core.memory.append((state, action, reward, next_state_arr, done))
                    core.optimize_model()
                    core.epsilon = max(config.EPSILON_MIN, core.epsilon * config.EPSILON_DECAY_STEP)

                total_reward += reward

                if done:
                    logger.debug(f"[{agent_name} | Игра {ep_num}] Конец. Шагов: {step} | Награда: {total_reward:.1f}")
                    break
                
                state = next_state

            if config.TRAIN_MODE and ep_num % config.TARGET_UPDATE == 0:
                core.update_target_network()
                logger.info("Целевая сеть обновлена.")

            if step == config.MAX_STEPS and not stop_event.is_set():
                await page.reload(wait_until="domcontentloaded")
                await asyncio.sleep(1.0)
                await page.mouse.click(config.WIN_W // 2, config.WIN_H // 2)

            stats = core.agent_stats[agent_name]
            stats["episodes_played"] += 1
            stats["total_reward"] += total_reward
            if step > stats["max_score"]:
                stats["max_score"] = step
                logger.warning(f"[{agent_name}] НОВЫЙ РЕКОРД: {step} очков!")
            if step < stats["min_score"]: stats["min_score"] = step

            stats["last_100_scores"].append(step)
            if len(stats["last_100_scores"]) > 100: stats["last_100_scores"].pop(0)
            stats["avg_last_100"] = round(sum(stats["last_100_scores"]) / len(stats["last_100_scores"]), 1)

            if not stop_event.is_set():
                try: await page.keyboard.up("ArrowDown")
                except: pass

            if config.TRAIN_MODE and not stop_event.is_set():
                if ep_num % 10 == 0: await core.save_all()

    except Exception as error:
        if not stop_event.is_set():
            logger.error(f"[{agent_name}] Ошибка: {error}")
        stop_event.set()