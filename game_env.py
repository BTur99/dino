# game_env.py
import asyncio
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import config


@dataclass
class Observation:
    state: Optional[Tuple]      # кортеж из STATE_DIM бинов или None
    status: str                 # OK / DEAD / NOT_PLAYING / PAUSED / NO_RUNNER / ERROR: ...
    jumping: bool
    ttc: float
    speed: float
    score: float                # отображаемый счёт = round(distanceRan * 0.025)
    distance: float = 0.0       # сырые пиксели distanceRan (для признаков/reward)


_GET_STATE_JS = """
() => {
    const runner = window.Runner?.instance_;
    if (!runner) return { status: "NO_RUNNER" };
    if (runner.crashed) return { status: "DEAD" };
    if (runner.paused) return { status: "PAUSED" };
    if (!runner.playing) return { status: "NOT_PLAYING" };

    const dino = runner.tRex;
    if (!dino) return { status: "NO_RUNNER" };
    const dinoWidth = dino.config ? dino.config.WIDTH : 44;
    const dinoRight = dino.xPos + dinoWidth;
    const obstacles = runner.horizon?.obstacles || [];

    const valid = obstacles
        .map(o => ({ ...o, dist: o.xPos - dinoRight }))
        .filter(o => o.dist > -o.width)
        .sort((a, b) => a.dist - b.dist);

    const best = valid[0];
    const second = valid[1];

    if (!best) {
        return {
            status: "OK", distance: 1000, speed: runner.currentSpeed,
            yPos: 100, jumping: dino.jumping, isBird: false, width: 20,
            secondDist: 1000, distanceRan: runner.distanceRan
        };
    }

    return {
        status: "OK", distance: best.dist, speed: runner.currentSpeed,
        yPos: best.yPos, jumping: dino.jumping,
        isBird: !!(best.typeConfig && best.typeConfig.type === "PTERODACTYL"),
        width: best.width,
        secondDist: second ? second.dist : 1000,
        distanceRan: runner.distanceRan
    };
}
"""


async def get_state(page) -> Observation:
    try:
        result = await asyncio.wait_for(page.evaluate(_GET_STATE_JS), timeout=2.0)
    except Exception as error:
        return Observation(None, f"ERROR: {error}", False, 0.0, 0.0, 0.0)

    status = result.get("status", "ERROR")
    if status != "OK":
        return Observation(None, status, False, 0.0, 0.0, 0.0)

    distance = result.get("distance")
    second_dist = result.get("secondDist", 1000)
    speed = result.get("speed")
    y_pos = result.get("yPos")
    raw_distance = result.get("distanceRan", 0.0)  # сырые пиксели пробега

    # NaN/Inf в любом из признаков превращается в ERROR, чтобы не отравить буфер.
    if not all(math.isfinite(v) for v in (distance, second_dist, speed, y_pos, raw_distance)):
        return Observation(None, "ERROR: non-finite state", False, 0.0, 0.0, 0.0)

    jumping = bool(result.get("jumping", False))
    is_bird = bool(result.get("isBird", False))
    width = result.get("width", 20)

    safe_speed = max(speed, 1.0)
    ttc = distance / safe_speed

    # Корзины TTC (в кадрах)
    if distance >= 500:
        dist_bin = 4
    elif ttc > 40:
        dist_bin = 0       # Слишком рано
    elif ttc > 25:
        dist_bin = 1       # Средне
    elif ttc > 15:
        dist_bin = 2       # Близко (идеально)
    else:
        dist_bin = 3       # Поздно

    second_ttc = second_dist / safe_speed
    if second_ttc > 45:
        second_bin = 0
    elif second_ttc > 25:
        second_bin = 1
    elif second_ttc > 15:
        second_bin = 2
    else:
        second_bin = 3

    if y_pos < 60:
        height_bin = 2
    elif y_pos < 90:
        height_bin = 1
    else:
        height_bin = 0

    bird_bin = 1 if is_bird else 0
    width_bin = 1 if width > 35 else 0

    state = (dist_bin, height_bin, bird_bin, width_bin, second_bin)
    # Отображаемый счёт — это distanceRan * 0.025, а не сырые пиксели.
    score = round(raw_distance * 0.025)
    return Observation(state, "OK", jumping, ttc, speed, score, raw_distance)


async def start_game(page) -> Observation:
    await page.keyboard.up("ArrowDown")
    await asyncio.sleep(0.85)
    obs = Observation(None, "NOT_PLAYING", False, 0.0, 0.0, 0.0)
    for _ in range(5):
        await page.keyboard.press("Space")
        await asyncio.sleep(0.15)
        obs = await get_state(page)
        if obs.state is not None:
            return obs
    return obs
