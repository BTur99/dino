# trainer.py
import asyncio
import config
import core
import game_env
from logger_setup import logger

async def worker(worker_id, agent_name, context, stop_event):
    try:
        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda: stop_event.set())
        
        # 1. Меняем заголовок вкладки браузера
        await page.add_init_script(f"document.title = '{agent_name}';")
        
        # 2. Загружаем игру
        await page.goto(config.GAME_URL, wait_until="networkidle")
        await asyncio.sleep(1.0)
        
        # 3. Рисуем имя по центру игры
        await page.evaluate(f"""
            () => {{
                const div = document.createElement('div');
                div.innerText = '{agent_name}';
                div.style.position = 'absolute';
                div.style.top = '100px';
                div.style.left = '50%';
                div.style.transform = 'translateX(-50%)';
                div.style.fontSize = '40px';
                div.style.fontWeight = 'bold';
                div.style.color = 'rgba(0, 0, 0, 0.15)';
                div.style.fontFamily = 'Arial, sans-serif';
                div.style.pointerEvents = 'none';
                div.style.zIndex = '9999';
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

            # Получаем стартовое состояние
            state, info, jumping, ttc, speed, status = await game_env.start_game(page)
            
            if stop_event.is_set():
                break

            if state is None:
                logger.warning(f"[{agent_name}] Игра не запустилась ({status}). Перезагрузка...")
                await page.reload(wait_until="networkidle")
                await asyncio.sleep(1.0)
                await page.mouse.click(config.WIN_W // 2, config.WIN_H // 2)
                continue

            total_reward = 0.0
            last_decision = None
            step = 0

            for step in range(1, config.MAX_STEPS + 1):
                if stop_event.is_set():
                    break
                    
                # --- ДЕЙСТВИЕ (теперь доступно и в воздухе) ---
                action = core.choose_action(state, current_eps)
                last_decision = (state, action)
                current_ttc = ttc
                current_speed = speed 

                if action == config.JUMP:
                    if not jumping:
                        await page.keyboard.press("Space")
                elif action == config.DUCK:
                    # Присед работает и на земле, и в воздухе (быстрое падение)
                    await page.keyboard.down("ArrowDown")
                else:
                    await page.keyboard.up("ArrowDown")

                await asyncio.sleep(config.STEP_DELAY)
                next_state, info, jumping, ttc, speed, status = await game_env.get_state(page)

                # --- СИСТЕМА НАГРАД (ИСПРАВЛЕНА) ---
                if next_state is None:
                    reward = -100.0 if status == "DEAD" else 0.0
                else:
                    if jumping:
                        # === ДИНОЗАВР В ВОЗДУХЕ ===
                        if action == 2: 
                            reward = config.DUCK_REWARD # +0.1 за быстрое падение
                        else:
                            reward = 0.0 # Просто летит — не получает ничего!
                    else:
                        # === ДИНОЗАВР НА ЗЕМЛЕ ===
                        if action == 1: # Прыжок
                            if current_ttc >= 20.0:
                                reward = -50.0  # Прыжок в пустоту
                            elif current_ttc > 15.0:
                                reward = -2.0   # Слишком ранний прыжок
                            else:
                                reward = 0.5    # Прыжок по делу
                        elif action == 0: # Бег
                            reward = 1.0 + (current_speed / 20.0)
                            if current_speed > config.HIGH_SPEED_THRESHOLD:
                                reward += config.HIGH_SPEED_BONUS
                        else: # Присед на земле
                            reward = config.DUCK_REWARD

                # --- ОБНОВЛЕНИЕ ЗНАНИЙ ---
                if config.TRAIN_MODE:
                    core.update_q_table(state, action, reward, next_state)
                    core.epsilon = max(config.EPSILON_MIN, core.epsilon * config.EPSILON_DECAY_STEP)

                total_reward += reward
                state = next_state

                # Если умерли — выходим из цикла шагов
                if next_state is None:
                    logger.debug(f"[{agent_name} | Игра {ep_num}] Конец ({status}). Шагов: {step} | Награда: {total_reward:.1f}")
                    break

            # Перезапуск игры, если агент пережил лимит шагов
            if step == config.MAX_STEPS and not stop_event.is_set():
                logger.info(f"[{agent_name} | Игра {ep_num}] ЛИМИТ ШАГОВ ДОСТИГНУТ ({config.MAX_STEPS}). Перезапуск игры.")
                await page.reload(wait_until="networkidle")
                await asyncio.sleep(1.0)
                await page.mouse.click(config.WIN_W // 2, config.WIN_H // 2)

            # --- ОБНОВЛЕНИЕ СТАТИСТИКИ ---
            stats = core.agent_stats[agent_name]
            stats["episodes_played"] += 1
            stats["total_reward"] += total_reward
            
            if step > stats["max_score"]:
                stats["max_score"] = step
                logger.warning(f"[{agent_name}] НОВЫЙ РЕКОРД: {step} очков!")
                
            if step < stats["min_score"]:
                stats["min_score"] = step

            stats["last_100_scores"].append(step)
            if len(stats["last_100_scores"]) > 100:
                stats["last_100_scores"].pop(0)
                
            stats["avg_last_100"] = round(sum(stats["last_100_scores"]) / len(stats["last_100_scores"]), 1)

            if not stop_event.is_set():
                try:
                    await page.keyboard.up("ArrowDown")
                except Exception:
                    pass

            # --- ЗАТУХАНИЕ АЛГОРИТМА И СОХРАНЕНИЕ ---
            if config.TRAIN_MODE and not stop_event.is_set():
                core.current_alpha = max(config.ALPHA_MIN, core.current_alpha * config.ALPHA_DECAY)
                
                if ep_num % 10 == 0:
                    await core.save_all()

    except Exception as error:
        if not stop_event.is_set():
            logger.error(f"[{agent_name}] Окно закрыто или произошла ошибка: {error}")
        stop_event.set()