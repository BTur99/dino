# core.py
import os
import json
import random
import asyncio
import copy
import numpy as np
import config
from logger_setup import logger  # ИМПОРТ ЛОГГЕРА

q_table = {}
episodes_done = 0
epsilon = config.EPSILON_START
current_alpha = config.ALPHA 
agent_stats = {}

for name in config.AGENT_NAMES:
    agent_stats[name] = {
        "max_score": 0,
        "min_score": 99999,
        "total_reward": 0.0,
        "episodes_played": 0,
        "last_100_scores": [],
        "avg_last_100": 0.0
    }

save_lock = asyncio.Lock()

if not os.path.exists("stats"):
    os.makedirs("stats")

def get_q_values(state):
    if state not in q_table:
        q_table[state] = [0.0] * len(config.ACTIONS)
    return q_table[state]

def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_all():
    global q_table, episodes_done, epsilon, current_alpha
    if config.LOAD_MODEL and os.path.exists(config.MODEL_FILE):
        q_table = np.load(config.MODEL_FILE, allow_pickle=True).item()
        logger.info(f"Загружена Q-таблица ({len(q_table)} состояний).")
    else:
        logger.info("Q-таблица создана с нуля.")
    
    if config.LOAD_MODEL and os.path.exists(config.STATE_FILE):
        with open(config.STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        episodes_done = int(data.get("episodes_done", 0))
        epsilon = float(data.get("epsilon", config.EPSILON_START))
        current_alpha = float(data.get("current_alpha", config.ALPHA))
        logger.info(f"Загружено состояние: эпизодов {episodes_done}, eps: {epsilon:.4f}, alpha: {current_alpha:.4f}")

async def save_all():
    async with save_lock:
        if not config.SAVE_MODEL:
            return
        await asyncio.to_thread(np.save, config.MODEL_FILE, q_table)
        state_data = {"episodes_done": episodes_done, "epsilon": epsilon, "current_alpha": current_alpha}
        await asyncio.to_thread(_write_json, config.STATE_FILE, state_data)
        
        stats_copy = copy.deepcopy(agent_stats)
        for name, stats in stats_copy.items():
            await asyncio.to_thread(_write_json, f"stats/stats_{name}.json", stats)
            
        logger.info(f"[Сохранение] эпизодов: {episodes_done} | eps: {epsilon:.4f} | alpha: {current_alpha:.4f}")

def choose_action(state, current_epsilon):
    values = get_q_values(state)
    if config.TRAIN_MODE and random.random() < current_epsilon:
        return random.choice(config.ACTIONS)
    return int(np.argmax(values))

def update_q_table(state, action, reward, next_state):
    values = get_q_values(state)
    current_q = values[action]
    max_next_q = 0.0 if next_state is None else max(get_q_values(next_state))
    values[action] = current_q + current_alpha * (reward + config.GAMMA * max_next_q - current_q)