# Ревью кода (2026-11-19)

Проверены: `main.py`, `trainer.py`, `core.py`, `game_env.py`, `config.py`,
`logger_setup.py`, `dashboard.py`, `clear.py`, `pyproject.toml`, `README.md`,
`.gitignore`, исходник игры `wayou/t-rex-runner/index.js`,
`playwright/_impl/event_emitter.py`.

Ограничения: shell в сессии ревьюера был недоступен (`git diff/status` не запускались);
чекпоинтов `dino_dqn.pth`/`state_v2.json` в дереве нет — содержимое проверить нечем.

## Топ-5

1. **`trainer.py:59-71` + `87-102`** — reward по состоянию ПОСЛЕ действия; каждый кадр
   в воздухе `-0.5`. `jumping` перезаписывается строкой 71 значением после нажатия,
   поэтому `action == JUMP` при `not jumping` почти всегда даёт `jumping == True` на
   следующем чтении → ветка `reward -= 0.5`, а не `+REWARD_GOOD_JUMP`. Прыжок ~35–40
   кадров → `-15…-20` за прыжок против `+0.25`. Оптимальная политика — не прыгать.
   Фикс: сохранить `current_jumping` до действия; `-0.5` вынести в константу.

2. **`trainer.py:63-68`** — в JUMP не отпускается `ArrowDown`. Игра разрешает прыжок
   только при `!jumping && !ducking`, а `ducking` снимается по keyup. «Присесть →
   прыгнуть» = no-op, но reward начисляется. Фикс: `keyboard.up("ArrowDown")` перед `Space`.

3. **`trainer.py:12` + `74-79` + `105`** — `page.on("close", lambda: stop_event.set())`
   не работает (слушатель вызывается с аргументом → `TypeError` внутри эмиттера).
   NOT_PLAYING/ERROR пишет мусор и падает с `NameError` (на первом шаге `next_state`
   не определён); воркер может до 20k шагов добавлять `(s, a, 0, s, False)`.
   Фикс: `lambda _: stop_event.set()`; NOT_PLAYING/ERROR → конец эпизода;
   `next_state = state` в начале; добавить `runner.paused` в статусы.

4. **`core.py:72-82` + `56-70`** — чекпоинт без optimizer/step/replay/target/stats;
   запись неатомарная (прямо в целевой файл, `torch.save` в потоке при активном
   `optimizer.step()` других воркеров); `load_all` без try. Resume стартует Adam с нуля
   и перезаписывает `stats` нулями. Фикс: словарь `{model, optimizer, step, epsilon,
   episodes_done, stats, target}`, запись через `*.tmp` + `os.replace`, снимок весов
   до `to_thread`.

5. **`core.py:93-112` + `trainer.py:104-108,118`** — `optimize_model()` из каждого из 20
   воркеров на каждом шаге, синхронно в общем event loop; `torch.set_num_threads` не
   задан. Target обновляется по **эпизодам** (`ep_num % TARGET_UPDATE`), при
   `TARGET_UPDATE=1000` и `TOTAL_EPISODES=6000` — 6 раз за весь запуск.
   Фикс: оптимизация по расписанию, `torch.set_num_threads(2)`, target по шагам/Polyak.

## Критично

| Место | Проблема | Фикс |
|---|---|---|
| `core.py:105-112` | нет `nan/inf`-проверок, нет `clip_grad_norm_` | `isfinite`, `clip_grad_norm_(..., 10)` |
| `game_env.py:53-60` | `speed/distance/yPos` не валидируются, `max(nan,1.0)→nan` | `math.isfinite` иначе `status="ERROR"` |
| `game_env.py:86-87` | `except Exception` → `"ERROR"` без завершения эпизода | ERROR завершает эпизод |
| `trainer.py:146-149` | одно исключение воркера ставит `stop_event` | рестарт на уровне воркера + счётчик ошибок |
| `game_env.py:7-45` | `page.evaluate` без таймаута | `asyncio.wait_for(get_state(...), 2.0)` |
| `main.py:94-98` | отменённые таски не `await`, bare `except` | `gather(..., return_exceptions=True)`, чистить `Singleton*` |

## Важно

- `main.py:33-34`: сиды после `import core`; `torch.manual_seed` нет; JS `Math.random` не сеется → недетерминизм.
- `trainer.py:45,108`: `epsilon` общий на 20 воркеров, читается раз на эпизод.
- `trainer.py:42-44`: глобальный `episodes_done` на всех воркеров; `TOTAL_EPISODES` общий.
- `trainer.py:127-137`: `max_score`/`avg_last_100` — шаги, а не очки игры.
- `core.py:84-91`: `random.choice`/`random.random` на общем RNG 20 воркеров — недетерминизм.
- `game_env.py:89-98`: `start_game` не проверяет реальный рестарт; при `NO_RUNNER` — цикл `continue` без лога.
- `trainer.py:19-20`: bare `except` ловит `CancelledError` → при Ctrl+C повторная навигация.
- `game_env.py:63-67`: `distance >= 500 → bin 4` рвёт TTC-окно на высоких скоростях.
- `trainer.py:70-71`: `sleep(0.015)` не синхронизирован с rAF (16.67 мс).
- `main.py:55-79`, `config.py:28`: `HEADLESS=False` → окна крадут фокус → blur → пауза.
- `core.py:37`: `deque` + `random.sample` — O(n) на сэмпл; PER нет.
- `core.py:106-107`: обычный DQN (`target.max`), не double.
- `main.py:15-23`: `import tkinter` на верхнем уровне — нужен `python3-tk`.
- `dashboard.py:43-104`: `while True + sleep(3)` в Streamlit — скрипт не завершается.
- `dashboard.py:32`, `core.py:50,53`: захардкожен `state_v2.json`; пути относительные.
- `core.py:58-70`: `LOAD_MODEL=True` без флага resume и без проверки архитектуры.

## Мелочи

- `pyproject.toml:20-21` + `src/dino/__init__.py`: entry point `dino` = «Hello from dino!», реальные модули не в пакете.
- `pyproject.toml:11-17`: `matplotlib`/`seaborn` не используются; torch с PyPI тянет CUDA-колёса (несколько ГБ).
- `README.md:1`: одна строка; нет инструкций по `uv sync`/`playwright install chrome`/`clear.py`.
- `.gitignore`: нет `*.pth`, `app.log`, `stats/`.
- `config.py`: `TARGET_UPDATE` в эпизодах; `STEP_DELAY=0.015` магическое; пути относительные.
- `main.py:26`: магическая `25`; `main.py:52`: `AGENT_NAMES[i]` падает при `NUM_WORKERS > 20`.
- `main.py:66-71`: `--window-position` по viewport, окна перекрываются.
- `core.py:31-32`, `trainer.py:105`: размерность `5` продублирована — вынести `STATE_DIM`.
- `trainer.py:61`: `state[2]` — магический индекс bird_bin.
- `trainer.py:113,132`: лог на каждый эпизод/рекорд; `app.log` без ротации.
- `clear.py:1`, `dashboard.py:1`: устаревшие заголовки; `clear.py:28` недостижимый `except`.
- `logger_setup.py:7`: DEBUG в файл; пересоздание хендлеров по `hasHandlers`.
- `dashboard.py:69-70`: `sort_values("avg_last_100")` упадёт на старом формате stats.
- Нет ни одного теста; `get_state` возвращает `status` дважды, `info` не используется.

## Что сделано хорошо

- Состояние из внутренних объектов игры (`Runner.instance_`, `horizon.obstacles`, `tRex`), а не из пикселей.
- Столкновение по `runner.crashed`, а не эвристике.
- TTC в кадрах (`distance/speed`) — инвариантно к скорости.
- Рестарт через собственную логику игры (keyup Space после `GAMEOVER_CLEAR_TIME`).
- `save_lock` + `asyncio.to_thread`, `deque(maxlen=…)`, защита `len(memory) < BATCH_SIZE`.
- Разделение headless/visible, persistent context на агента, подпись окна именем.
- Компактная модель (5 входов) — на CPU обучается за секунды после фикса reward.

## Чеклист перед первым длинным запуском

1. `current_jumping` до действия; убрать `-0.5` в константу; проверить, что успешный прыжок даёт `+`.
2. JUMP: отпустить `ArrowDown` перед `Space`; вручную проверить «присел → прыгнул».
3. NOT_PLAYING/ERROR → конец эпизода; `next_state = state`; `lambda _: stop_event.set()`.
4. Чекпоинт: optimizer/step/stats/атомарность; проверить resume после Ctrl+C.
5. Оптимизация по расписанию; `torch.set_num_threads(2)`; target по шагам.
6. `isfinite` + `clip_grad_norm_`; лог loss; loss не NaN после 10 минут.
7. `torch.manual_seed` до сетей; пересеять JS; `HEADLESS=True`; smoke 1–2 минуты.
