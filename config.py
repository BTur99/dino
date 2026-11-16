GAME_URL = "https://wayou.github.io/t-rex-runner/"

# Настройки обучения
NUM_WORKERS = 10         
TOTAL_EPISODES = 3000    
ALPHA = 0.15
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01           
EPSILON_DECAY_STEP = 0.9999  

STEP_DELAY = 0.04            
MAX_STEPS = 3000
MODEL_FILE = "q_table_v2.npy"
STATE_FILE = "state_v2.json"

TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True
HEADLESS = False
SEED = 42

# Окна
WIN_W = 650
WIN_H = 350
OFFSET_X = 350

# Имена и действия
AGENT_NAMES = [
    "Андрей", "Тимур", "Иван", "Далер", "Платон", 
    "Александр", "Тимофей", "Матвей", "Ярик", "Паша"
]

ACTIONS = [0, 1, 2]
NOOP, JUMP, DUCK = 0, 1, 2
