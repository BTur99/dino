# config.py
# GAME_URL = "http://localhost:8000/"
GAME_URL = "https://wayou.github.io/t-rex-runner/"

NUM_WORKERS = 20         
VISIBLE_WORKERS = 4      
TOTAL_EPISODES = 6000    

# --- НАСТРОЙКИ DQN (Нейросети) ---
GAMMA = 0.99             
LR = 0.0001              # Уменьшили, чтобы сеть не сходила с ума
BATCH_SIZE = 128          
MEMORY_SIZE = 250000      # Увеличили память для 20 агентов
TARGET_UPDATE = 1000      
EPSILON_START = 1.0
EPSILON_MIN = 0.02           
EPSILON_DECAY_STEP = 0.99999 # Очень медленное затухание

STEP_DELAY = 0.015        
MAX_STEPS = 20000

MODEL_FILE = "dino_dqn.pth" 
STATE_FILE = "state_v2.json"

TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True
HEADLESS = False         
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

# --- МАТЕМАТИЧЕСКИ СТРОГАЯ СИСТЕМА НАГРАД ---
REWARD_ALIVE = 0.005         # Оптимальный базовый поток наград (Q_max ≈ +0.5 при gamma=0.99)
REWARD_DEATH = -1.0          # Эталонный штраф для стабильности функции потерь (Huber/MSE)
REWARD_SPEED_BONUS = 0.0005  # Плавный рост ценности выживания на высоких скоростях
REWARD_GOOD_JUMP = 0.25      # Четкий позитивный сигнал за преодоление препятствия
REWARD_EARLY_JUMP = -0.15    # Надежный фильтр от случайного спама пробелом
REWARD_LATE_JUMP = -0.01     # Минимальный сигнал, не искажающий TD-ошибку перед смертью
REWARD_USELESS_DUCK = -0.005 # Сверхмягкий штраф: не убивает желание приседать при появлении птиц
REWARD_FAST_FALL = 0.005     # Безопасный микро-бонус (не перекрывает штраф за ранний прыжок)
