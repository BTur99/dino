# core.py
import os
import json
import random
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
from logger_setup import logger

torch.set_num_threads(2)

# Сиды ДО создания сетей, иначе структура/инициализация весов недетерминирована.
torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используемое устройство: {device}")


class DinoNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DinoNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SumTree:
    """Бинарное дерево сумм приоритетов: листья на [capacity-1 .. 2*capacity-2],
    корень — сумма всех приоритетов. Сэмплинг и обновление за O(log n)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0          # позиция следующей записи (кольцевой буфер)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, idx, value):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        right = left + 1
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    def get(self, value):
        """Возвращает (индекс листа, приоритет, переход) для случайного значения value."""
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, beta_start, beta_steps, eps):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / max(beta_steps, 1)
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.n_entries

    def push(self, transition):
        # Новые переходы получают максимальный приоритет, чтобы гарантированно
        # быть сэмплированными хотя бы раз.
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        n = len(self)
        self.beta = min(1.0, self.beta + self.beta_increment)

        indices = []
        transitions = []
        weights = []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            value = random.uniform(segment * i, segment * (i + 1))
            idx, priority, transition = self.tree.get(value)
            indices.append(idx)
            transitions.append(transition)
            prob = priority / self.tree.total
            weights.append((n * prob) ** (-self.beta))

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()  # нормализуем к max, чтобы не менять масштаб градиентов
        return indices, transitions, weights

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)


policy_net = DinoNet(config.STATE_DIM, len(config.ACTIONS)).to(device)
target_net = DinoNet(config.STATE_DIM, len(config.ACTIONS)).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=config.LR)
logger.info(
    f"Сеть: вход {config.STATE_DIM} признаков → 128 → 128 → "
    f"{len(config.ACTIONS)} действий (PER, MEMORY_SIZE={config.MEMORY_SIZE})"
)
memory = PrioritizedReplayBuffer(
    config.MEMORY_SIZE,
    alpha=config.PER_ALPHA,
    beta_start=config.PER_BETA_START,
    beta_steps=config.PER_BETA_STEPS,
    eps=config.PER_EPS,
)

episodes_done = 0
steps_done = 0
epsilon = config.EPSILON_START
agent_stats = {}

for name in config.AGENT_NAMES:
    agent_stats[name] = {
        "max_score": 0, "min_score": 99999, "total_reward": 0.0,
        "episodes_played": 0, "last_100_scores": [], "avg_last_100": 0.0
    }

save_lock = asyncio.Lock()
if not os.path.exists("stats"):
    os.makedirs("stats")


def _write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


def _clone_tensors(obj):
    """Рекурсивно переносим тензоры на CPU и клонируем (снимок для сохранения)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _clone_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clone_tensors(v) for v in obj]
    return obj


def _snapshot():
    return {
        "model": _clone_tensors(policy_net.state_dict()),
        "target": _clone_tensors(target_net.state_dict()),
        "optimizer": _clone_tensors(optimizer.state_dict()),
        "steps_done": steps_done,
        "episodes_done": episodes_done,
        "epsilon": epsilon,
        "stats": json.loads(json.dumps(agent_stats)),
    }


def load_all():
    global episodes_done, steps_done, epsilon
    if not config.LOAD_MODEL:
        logger.info("Загрузка отключена (LOAD_MODEL=False). Обучение с нуля.")
        return
    if not os.path.exists(config.MODEL_FILE):
        logger.info("Чекпоинт не найден. Обучение с нуля.")
        return

    try:
        ckpt = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)
    except Exception as error:
        logger.error(f"Не удалось прочитать чекпоинт {config.MODEL_FILE}: {error}")
        return

    try:
        if isinstance(ckpt, dict) and "model" in ckpt:
            # Новый формат — полный чекпоинт.
            policy_net.load_state_dict(ckpt["model"])
            target_net.load_state_dict(ckpt.get("target", ckpt["model"]))
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            steps_done = int(ckpt.get("steps_done", 0))
            episodes_done = int(ckpt.get("episodes_done", 0))
            epsilon = float(ckpt.get("epsilon", config.EPSILON_START))
            saved_stats = ckpt.get("stats", {})
            for name, stats in saved_stats.items():
                if name in agent_stats:
                    agent_stats[name] = {**agent_stats[name], **stats}
        else:
            # Старый формат — только веса политики.
            policy_net.load_state_dict(ckpt)
            target_net.load_state_dict(ckpt)
            # ep/eps хранились отдельно в лёгком STATE_FILE — восстанавливаем их,
            # иначе epsilon сбросится на 1.0 и потеряется прогресс эпизодов.
            if os.path.exists(config.STATE_FILE):
                try:
                    with open(config.STATE_FILE, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                    episodes_done = int(state_data.get("episodes_done", 0))
                    epsilon = float(state_data.get("epsilon", config.EPSILON_START))
                except Exception as error:
                    logger.warning(f"Не удалось прочитать {config.STATE_FILE}: {error}")
        target_net.eval()
        logger.info(
            f"Чекпоинт загружен: эпизодов {episodes_done}, шагов {steps_done}, eps {epsilon:.4f}"
        )
    except Exception as error:
        logger.error(f"Ошибка восстановления чекпоинта (продолжаем с нуля): {error}")


async def save_all():
    async with save_lock:
        if not config.SAVE_MODEL:
            return

        snapshot = _snapshot()  # снимок ДО to_thread, чтобы не сохранить «живые» тензоры
        tmp = config.MODEL_FILE + ".tmp"

        def _write_model():
            torch.save(snapshot, tmp)
            os.replace(tmp, config.MODEL_FILE)

        await asyncio.to_thread(_write_model)

        state_data = {
            "episodes_done": episodes_done, "steps_done": steps_done, "epsilon": epsilon
        }
        await asyncio.to_thread(_write_json_atomic, config.STATE_FILE, state_data)

        for name, stats in agent_stats.items():
            await asyncio.to_thread(
                _write_json_atomic, f"stats/stats_{name}.json", dict(stats)
            )
        logger.info(
            f"[Сохранение] эпизодов: {episodes_done} | шагов: {steps_done} | eps: {epsilon:.4f}"
        )


def choose_action(state, current_epsilon):
    if config.TRAIN_MODE and random.random() < current_epsilon:
        return random.choice(config.ACTIONS)

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax().item())


def optimize_model():
    if len(memory) < config.BATCH_SIZE:
        return None

    indices, batch, weights = memory.sample(config.BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(np.array(state_batch, dtype=np.float32)).to(device)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(np.array(next_state_batch, dtype=np.float32)).to(device)
    done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
    weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

    # NaN/Inf в батче убьёт веса навсегда — пропускаем такой батч.
    if not (torch.isfinite(state_batch).all()
            and torch.isfinite(next_state_batch).all()
            and torch.isfinite(reward_batch).all()):
        return None

    q_values = policy_net(state_batch).gather(1, action_batch)

    # Double DQN: действие выбираем по policy_net, значение берём из target_net.
    with torch.no_grad():
        next_actions = policy_net(next_state_batch).argmax(1).unsqueeze(1)
        next_q_values = target_net(next_state_batch).gather(1, next_actions)
        expected_q_values = reward_batch + config.GAMMA * next_q_values * (1.0 - done_batch)

    # PER: TD-ошибки нужны для обновления приоритетов, а IS-веса — для несмещённой оценки.
    td_errors = (q_values - expected_q_values).detach()
    loss = (weights * F.smooth_l1_loss(q_values, expected_q_values, reduction="none")).mean()

    # Проверяем ДО обратного прохода: NaN/Inf в loss (а значит и в градиентах)
    # не должен успевать уйти в веса через optimizer.step().
    if not torch.isfinite(loss):
        return None

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    memory.update_priorities(indices, td_errors.cpu().numpy().squeeze(1))

    return float(loss.detach().cpu().item())


def record_transition(state, action, reward, next_state, done):
    """Общая точка записи перехода: счётчик шагов, оптимизация по расписанию
    и обновление target-сети по счётчику шагов. Epsilon обновляется в trainer.py
    один раз в начале эпизода (линейно по эпизодам)."""
    global steps_done
    memory.push((state, action, reward, next_state, done))
    steps_done += 1

    if steps_done % config.OPTIMIZE_EVERY == 0:
        loss = optimize_model()
        if loss is not None and steps_done % config.LOG_LOSS_EVERY == 0:
            logger.info(
                f"[шаг {steps_done}] loss={loss:.5f} | eps={epsilon:.4f} | "
                f"эпизодов={episodes_done}"
            )

    if steps_done % config.TARGET_UPDATE == 0 and steps_done > 0:
        update_target_network()


def update_target_network():
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    logger.info(f"[шаг {steps_done}] Целевая сеть обновлена.")
