# game_env.py
import asyncio
import config

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
                        secondDist: 1000, dinoY: dino.yPos // ДОБАВЛЕНО dinoY
                    };
                }

                return {
                    status: "OK", distance: best.dist, speed: runner.currentSpeed,
                    yPos: best.yPos, jumping: dino.jumping,
                    isBird: best.type === "PTERODACTYL" || best.isBird === true,
                    width: best.width,
                    secondDist: second ? second.dist : 1000,
                    dinoY: dino.yPos // ДОБАВЛЕНО dinoY
                };
            }
            """
        )

        status = result["status"]
        if status != "OK":
            return None, status, False, 0, 0, status

        distance = result["distance"]
        second_dist = result.get("secondDist", 1000)
        speed = result["speed"]
        y_pos = result["yPos"]
        jumping = result["jumping"]
        is_bird = result.get("isBird", False)
        width = result.get("width", 20)
        dino_y = result.get("dinoY", 0)
        
        safe_speed = max(speed, 1.0)
        ttc = distance / safe_speed

        if distance >= 500: dist_bin = 5
        elif ttc > 15.0: dist_bin = 0
        elif ttc > 10.0: dist_bin = 1
        elif ttc > 7.0: dist_bin = 2
        elif ttc > 5.0: dist_bin = 3
        else: dist_bin = 4

        second_ttc = second_dist / safe_speed
        if second_ttc > 15.0: second_bin = 0
        elif second_ttc > 10.0: second_bin = 1
        elif second_ttc > 6.0: second_bin = 2
        else: second_bin = 3

        if y_pos < 60: height_bin = 2  
        elif y_pos < 90: height_bin = 1  
        else: height_bin = 0  

        bird_bin = 1 if is_bird else 0
        width_bin = 1 if width > 35 else 0

        # НОВОЕ: Определяем высоту самого динозавра
        if dino_y == 0:
            dino_height_bin = 0 # На земле
        elif dino_y < 40:
            dino_height_bin = 1 # Низко (только что прыгнул или быстро падает)
        else:
            dino_height_bin = 2 # Высоко в воздухе

        # Состояние теперь состоит из 6 цифр!
        state = (dist_bin, height_bin, bird_bin, width_bin, second_bin, dino_height_bin)
        
        # Возвращаем speed для награды
        return state, "OK", jumping, ttc, speed, "OK"

    except Exception as error:
        return None, f"ERROR: {error}", False, 0, 0, "ERROR"

async def start_game(page):
    await page.keyboard.up("ArrowDown")
    await asyncio.sleep(0.85)

    for _ in range(5):
        await page.keyboard.press("Space")
        await asyncio.sleep(0.15)

        state, info, jumping, ttc, speed, status = await get_state(page)
        if state is not None:
            return state, info, jumping, ttc, speed, status

    return None, info, False, 0, 0, status