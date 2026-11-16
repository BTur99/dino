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
                        status: "OK", distance: 1000, speed: runner.currentSpeed,
                        yPos: 100, jumping: dino.jumping, isBird: false, width: 20
                    };
                }

                return {
                    status: "OK", distance: bestDist, speed: runner.currentSpeed,
                    yPos: best.yPos, jumping: dino.jumping,
                    isBird: best.type === "PTERODACTYL" || best.isBird === true,
                    width: best.width
                };
            }
            """
        )

        status = result["status"]
        if status != "OK":
            return None, status, False, 0, status

        distance = result["distance"]
        y_pos = result["yPos"]
        jumping = result["jumping"]
        is_bird = result.get("isBird", False)
        width = result.get("width", 20)
        
        if distance >= 500: dist_bin = 4  
        elif distance > 220: dist_bin = 0
        elif distance > 130: dist_bin = 1
        elif distance > 55: dist_bin = 2
        else: dist_bin = 3

        if y_pos < 60: height_bin = 2  
        elif y_pos < 90: height_bin = 1  
        else: height_bin = 0  

        bird_bin = 1 if is_bird else 0
        width_bin = 1 if width > 35 else 0

        state = (dist_bin, height_bin, bird_bin, width_bin)
        return state, "OK", jumping, distance, "OK"

    except Exception as error:
        return None, f"ERROR: {error}", False, 0, "ERROR"

async def start_game(page):
    await page.keyboard.up("ArrowDown")
    await asyncio.sleep(0.85)

    for _ in range(5):
        await page.keyboard.press("Space")
        await asyncio.sleep(0.15)

        state, info, jumping, dist, status = await get_state(page)
        if state is not None:
            return state, info, jumping, dist, status

    return None, info, False, 0, status