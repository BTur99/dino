# config.py
GAME_URL = "https://wayou.github.io/t-rex-runner/"

# --- Настройки обучения ---
NUM_WORKERS = 10         
TOTAL_EPISODES = 6000    # <--- УВЕЛИЧЕНО до 6000 для изучения новых 864 состояний

ALPHA = 0.15
ALPHA_MIN = 0.01         
ALPHA_DECAY = 0.9999     # Замедлили затухание, чтобы он успел выучить новые фишки (было 0.9995)

GAMMA = 0.99             

EPSILON_START = 1.0
EPSILON_MIN = 0.01           
EPSILON_DECAY_STEP = 0.999995 # Замедлили, чтобы он дольше исследовал новые комбинации (было 0.99995)

STEP_DELAY = 0.04            
MAX_STEPS = 10000

MODEL_FILE = "q_table_v2.npy"
STATE_FILE = "state_v2.json"

TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True
HEADLESS = False
SEED = 42

# --- Настройки окон ---
WIN_W = 650
WIN_H = 350
OFFSET_X = 350

# --- Имена и действия ---
AGENT_NAMES = [
    "Андрей", "Тимур", "Иван", "Далер", "Платон", 
    "Александр", "Тимофей", "Матвей", "Ярик", "Паша"
]

ACTIONS = [0, 1, 2]
NOOP, JUMP, DUCK = 0, 1, 2

# --- Настройки наград ---
HIGH_SPEED_THRESHOLD = 12.0  # Порог скорости для бонуса
HIGH_SPEED_BONUS = 0.5       # Дополнительный бонус за риск
DUCK_REWARD = 0.1            # Награда за присед (чтобы не злоупотреблял)