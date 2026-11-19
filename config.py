# config.py
GAME_URL = "http://127.0.0.1:8000/"

NUM_WORKERS = 10
# NUM_WORKERS = 8     # более консервативный вариант для слабых машин
# NUM_WORKERS = 12    # более агрессивный вариант
VISIBLE_WORKERS = 4
TOTAL_EPISODES = 6000

# --- НАСТРОЙКИ DQN (Нейросети) ---
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 128
MEMORY_SIZE = 250000
TARGET_UPDATE = 1000          # обновление target-сети раз в N ШАГОВ (не эпизодов!)
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY_STEP = 1.0

OPTIMIZE_EVERY = 4            # вызывать optimize_model() раз в N шагов (не каждый шаг)
LOG_LOSS_EVERY = 200          # логировать loss раз в N шагов

STEP_DELAY = 0.015
MAX_STEPS = 20000

MODEL_FILE = "dino_dqn.pth"   # полный чекпоинт: model, target, optimizer, шаги, eps, stats
STATE_FILE = "state_v2.json"  # лёгкое состояние для дашборда (episodes_done, epsilon)

TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True
HEADLESS = True
SEED = 42

WIN_W = 650
WIN_H = 350
OFFSET_X = 350

AGENT_NAMES = [
    "Андрей", "Тимур", "Иван", "Далер", "Платон",
    "Александр", "Тимофей", "Матвей", "Ярик", "Паша",
    "Егор", "Артем", "Роман", "Никита", "Максим",
    "Владимир", "Денис", "Евгений", "Кирилл", "Глеб"
]

ACTIONS = [0, 1, 2]
NOOP, JUMP, DUCK = 0, 1, 2

# Размерность состояния (5 корзин: дистанция, высота, птица, ширина, второе препятствие)
STATE_DIM = 5

# --- СИСТЕМА НАГРАД ---
REWARD_ALIVE = 0.005         # базовый поток наград за выживание
REWARD_DEATH = -1.0          # штраф за столкновение
REWARD_SPEED_BONUS = 0.0005  # плавный рост ценности выживания на высоких скоростях
REWARD_GOOD_JUMP = 0.25      # чёткий позитивный сигнал за преодоление препятствия
REWARD_EARLY_JUMP = -0.15    # фильтр от случайного спама пробелом (нет препятствия рядом)
REWARD_LATE_JUMP = -0.01     # минимальный сигнал, не искажающий TD-ошибку перед смертью
REWARD_USELESS_DUCK = -0.005 # сверхмягкий штраф за приседание без птицы
REWARD_FAST_FALL = 0.005     # микро-бонус за быстрое снижение (duck в воздухе)
# Штраф за кадр в воздухе. По модулю СТРОГО меньше REWARD_GOOD_JUMP, чтобы прыжок через
# кактус (~35 кадров в воздухе) в сумме давал положительную награду: 0.25 - 35*0.005 = +0.075.
REWARD_AIR_PENALTY = -0.005
