# game_env.py
import asyncio
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import config


@dataclass
class Observation:
    state: Optional[Tuple]      # кортеж из STATE_DIM непрерывных признаков или None
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
            yPos: 100, dinoY: dino.yPos, jumping: dino.jumping, isBird: false, width: 20,
            secondDist: 1000, distanceRan: runner.distanceRan
        };
    }

    return {
        status: "OK", distance: best.dist, speed: runner.currentSpeed,
        yPos: best.yPos, dinoY: dino.yPos, jumping: dino.jumping,
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
    dino_y = result.get("dinoY", 0.0)
    raw_distance = result.get("distanceRan", 0.0)  # сырые пиксели пробега

    # NaN/Inf в любом из признаков превращается в ERROR, чтобы не отравить буфер.
    if not all(math.isfinite(v) for v in (distance, second_dist, speed, y_pos, dino_y, raw_distance)):
        return Observation(None, "ERROR: non-finite state", False, 0.0, 0.0, 0.0)

    jumping = bool(result.get("jumping", False))
    is_bird = bool(result.get("isBird", False))
    width = result.get("width", 20)

    safe_speed = max(speed, 1.0)
    ttc = distance / safe_speed
    second_ttc = second_dist / safe_speed

    def _clip(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    # 8 непрерывных нормализованных признаков (все в диапазоне [0, 1]).
    state = (
        _clip(ttc / 60),                 # 0: время до столкновения (кадры) / 60
        _clip(distance / 600),           # 1: сырая дистанция до препятствия / 600
        1.0 if is_bird else 0.0,         # 2: тип препятствия — птица (PTERODACTYL)
        _clip(y_pos / 100),              # 3: высота препятствия (для duck vs jump)
        _clip(width / 60),               # 4: ширина препятствия / группы
        _clip(second_ttc / 60),          # 5: ttc второго препятствия
        _clip(speed / 13),               # 6: текущая скорость игры
        _clip(dino_y / 60),              # 7: высота динозавра (тайминг прыжка)
    )
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
