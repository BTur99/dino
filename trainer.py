# trainer.py
import asyncio
import math
import numpy as np
import config
import core
import game_env
from logger_setup import logger


def _prng_script(worker_id):
    """Детерминированный PRNG (mulberry32), чтобы пересеять Math.random в игре."""
    seed = (config.SEED * 100000 + worker_id) & 0xFFFFFFFF
    return f"""
    (() => {{
        let s = {seed} >>> 0;
        const next = () => {{
            s = (s + 0x6D2B79F5) >>> 0;
            let t = s;
            t = Math.imul(t ^ (t >>> 15), 1 | t);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        }};
        Math.random = next;
    }})();
    """


def _compute_reward(current_jumping, action, current_ttc, current_bird_bin, current_speed):
    """Награда на основе состояния ДО действия (current_jumping — до нажатия клавиш)."""
    reward = config.REWARD_ALIVE + current_speed * config.REWARD_SPEED_BONUS
    event = None

    if current_jumping:
        if action == config.DUCK:
            reward += config.REWARD_FAST_FALL
            event = "fast_fall"
        else:
            reward += config.REWARD_AIR_PENALTY
            event = "air"
    else:
        if action == config.JUMP:
            if current_ttc > 40:
                reward += config.REWARD_EARLY_JUMP
                event = "early_jump"
            elif current_ttc < 15:
                reward += config.REWARD_LATE_JUMP
                event = "late_jump"
            else:
                reward += config.REWARD_GOOD_JUMP
                event = "good_jump"
        elif action == config.DUCK:
            if current_bird_bin == 0:
                reward += config.REWARD_USELESS_DUCK
                event = "useless_duck"

    return reward, event


async def apply_action(page, action, current_jumping):
    """Единое управление клавишами: явно держим/отпускаем ArrowDown."""
    if action == config.JUMP:
        # Игра разрешает прыжок только из состояния без ducking.
        await page.keyboard.up("ArrowDown")
        if not current_jumping:
            await page.keyboard.press("Space")
    elif action == config.DUCK:
        await page.keyboard.down("ArrowDown")
    else:  # NOOP
        await page.keyboard.up("ArrowDown")


async def _play_episode(page, worker_id, agent_name, stop_event):
    core.episodes_done += 1
    ep_num = core.episodes_done
    current_eps = core.epsilon if config.TRAIN_MODE else 0.0

    obs = await game_env.start_game(page)
    if obs.state is None:
        if str(obs.status).startswith("ERROR"):
            raise RuntimeError(f"get_state сбой при запуске: {obs.status}")
        # NOT_PLAYING / NO_RUNNER / PAUSED — пробуем перезагрузить один раз.
        try:
            await page.reload(wait_until="domcontentloaded")
            await asyncio.sleep(1.0)
            obs = await game_env.start_game(page)
        except Exception:
            pass
        if obs.state is None:
            logger.warning(f"[{agent_name}] Не удалось запустить игру (эпизод {ep_num}): {obs.status}")
            return

    if stop_event.is_set():
        return

    state = np.array(obs.state, dtype=np.float32)
    jumping = obs.jumping
    ttc = obs.ttc
    speed = obs.speed
    score = obs.score
    total_reward = 0.0
    good_jumps = 0
    saw_bird = False
    next_state = state  # инициализация, чтобы не было NameError при раннем выходе
    last_step = 0

    for last_step in range(1, config.MAX_STEPS + 1):
        if stop_event.is_set():
            break

        action = core.choose_action(state, current_eps)
        current_ttc = ttc
        current_speed = speed
        current_bird_bin = int(state[2])
        current_jumping = jumping  # состояние ДО действия

        await apply_action(page, action, current_jumping)
        await asyncio.sleep(config.STEP_DELAY)

        obs = await game_env.get_state(page)

        if obs.state is None:
            if obs.status == "DEAD":
                reward = config.REWARD_DEATH
                done = True
                next_state = np.zeros(config.STATE_DIM, dtype=np.float32)
            else:
                # NOT_PLAYING / PAUSED / NO_RUNNER / ERROR — конец эпизода БЕЗ записи перехода.
                logger.debug(f"[{agent_name}] Эпизод {ep_num} прерван: {obs.status}")
                break
        else:
            next_state = np.array(obs.state, dtype=np.float32)
            reward, event = _compute_reward(
                current_jumping, action, current_ttc, current_bird_bin, current_speed
            )
            if event == "good_jump":
                good_jumps += 1
                logger.debug(
                    f"[{agent_name}] GOOD_JUMP +{config.REWARD_GOOD_JUMP} (ttc={current_ttc:.1f})"
                )
            if int(next_state[2]) == 1:
                saw_bird = True
            done = False

        if not math.isfinite(reward):
            reward = 0.0

        if config.TRAIN_MODE:
            core.record_transition(state, action, reward, next_state, done)

        total_reward += reward

        if done:
            logger.debug(
                f"[{agent_name} | Игра {ep_num}] Конец. Шагов: {last_step} | "
                f"Награда: {total_reward:.2f} | Счёт: {score:.0f}"
            )
            break

        state = next_state
        jumping = obs.jumping
        ttc = obs.ttc
        speed = obs.speed
        score = obs.score

    # --- Статистика (score — отображаемый счёт, уже в очках) ---
    stats = core.agent_stats[agent_name]
    stats["episodes_played"] += 1
    stats["total_reward"] += total_reward
    if score > stats["max_score"]:
        stats["max_score"] = score
        logger.info(f"[{agent_name}] НОВЫЙ РЕКОРД: {score:.0f} очков (эпизод {ep_num})")
    if score < stats["min_score"]:
        stats["min_score"] = score
    stats["last_100_scores"].append(score)
    if len(stats["last_100_scores"]) > 100:
        stats["last_100_scores"].pop(0)
    stats["avg_last_100"] = round(
        sum(stats["last_100_scores"]) / len(stats["last_100_scores"]), 1
    )

    logger.info(
        f"[{agent_name}] эп={ep_num} счёт={score:.0f} шагов={last_step} "
        f"reward={total_reward:.2f} good_jumps={good_jumps} bird={saw_bird}"
    )

    if not stop_event.is_set():
        try:
            await page.keyboard.up("ArrowDown")
        except Exception:
            pass

    # При выживании до MAX_STEPS перезагружаем страницу, чтобы начать чистую игру.
    if last_step == config.MAX_STEPS and not stop_event.is_set():
        try:
            await page.reload(wait_until="domcontentloaded")
            await asyncio.sleep(1.0)
        except Exception:
            pass

    if config.TRAIN_MODE and not stop_event.is_set() and ep_num % 10 == 0:
        await core.save_all()


async def worker(worker_id, agent_name, context, stop_event):
    page = None
    try:
        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda _: stop_event.set())
        await page.add_init_script(f"document.title = '{agent_name}';")
        await page.add_init_script(_prng_script(worker_id))

        # Навигация с повторной попыткой (локальный сервер может «подтормаживать»).
        for timeout in (10000, 15000):
            try:
                await page.goto(config.GAME_URL, wait_until="domcontentloaded", timeout=timeout)
                break
            except Exception:
                await asyncio.sleep(1.5)

        await asyncio.sleep(1.0)

        # Fail-fast: если сервер не поднят, page.url останется about:blank, а
        # get_state вернёт NO_RUNNER — иначе эпизоды «играются» впустую и
        # перезаписывают чекпоинт. Останавливаем запуск с явной подсказкой.
        obs = await game_env.get_state(page)
        if obs.status == "NO_RUNNER" or not page.url.startswith(config.GAME_URL):
            logger.error(
                "Сервер недоступен, запусти: python3 -m http.server 8000 "
                "--bind 127.0.0.1 --directory ~/Проекты/t-rex-runner"
            )
            stop_event.set()
            return

        if worker_id <= config.VISIBLE_WORKERS and not config.HEADLESS:
            await page.evaluate(
                f"""
                () => {{
                    const div = document.createElement('div');
                    div.innerText = '{agent_name}';
                    div.style.position = 'absolute'; div.style.top = '100px'; div.style.left = '50%';
                    div.style.transform = 'translateX(-50%)'; div.style.fontSize = '40px';
                    div.style.fontWeight = 'bold'; div.style.color = 'rgba(0, 0, 0, 0.15)';
                    div.style.pointerEvents = 'none'; div.style.zIndex = '9999';
                    document.body.appendChild(div);
                }}
                """
            )
            await page.bring_to_front()
            await page.mouse.click(config.WIN_W // 2, config.WIN_H // 2)

        await asyncio.sleep(0.5)

        while core.episodes_done < config.TOTAL_EPISODES and not stop_event.is_set():
            try:
                await _play_episode(page, worker_id, agent_name, stop_event)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(f"[{agent_name}] Ошибка эпизода, перезагружаю страницу: {error}")
                try:
                    await page.reload(wait_until="domcontentloaded")
                    await asyncio.sleep(1.0)
                except Exception:
                    raise  # страница мертва — воркер завершается

    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.error(f"[{agent_name}] Воркер завершён: {error}")
    # ВАЖНО: не ставим stop_event — один упавший воркер не должен останавливать весь запуск.
