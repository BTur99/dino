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
from collections import deque
import config
from logger_setup import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используемое устройство: {device}")

class DinoNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DinoNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ВХОДНЫЕ ДАННЫЕ ТЕПЕРЬ 5 (без dino_y)
policy_net = DinoNet(5, len(config.ACTIONS)).to(device)
target_net = DinoNet(5, len(config.ACTIONS)).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=config.LR)
memory = deque(maxlen=config.MEMORY_SIZE)

episodes_done = 0
epsilon = config.EPSILON_START
agent_stats = {}

for name in config.AGENT_NAMES:
    agent_stats[name] = {
        "max_score": 0, "min_score": 99999, "total_reward": 0.0,
        "episodes_played": 0, "last_100_scores": [], "avg_last_100": 0.0
    }

save_lock = asyncio.Lock()
if not os.path.exists("stats"): os.makedirs("stats")

def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_all():
    global episodes_done, epsilon
    if config.LOAD_MODEL and os.path.exists(config.MODEL_FILE):
        policy_net.load_state_dict(torch.load(config.MODEL_FILE, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        logger.info(f"Загружены веса нейросети.")
    else:
        logger.info("Нейросеть создана с нуля.")
    
    if config.LOAD_MODEL and os.path.exists(config.STATE_FILE):
        with open(config.STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        episodes_done = int(data.get("episodes_done", 0))
        epsilon = float(data.get("epsilon", config.EPSILON_START))
        logger.info(f"Загружено: эпизодов {episodes_done}, eps: {epsilon:.4f}")

async def save_all():
    async with save_lock:
        if not config.SAVE_MODEL: return
        await asyncio.to_thread(torch.save, policy_net.state_dict(), config.MODEL_FILE)
        state_data = {"episodes_done": episodes_done, "epsilon": epsilon}
        await asyncio.to_thread(_write_json, config.STATE_FILE, state_data)
        
        stats_copy = {k: v.copy() for k, v in agent_stats.items()}
        for name, stats in stats_copy.items():
            await asyncio.to_thread(_write_json, f"stats/stats_{name}.json", stats)
        logger.info(f"[Сохранение] эпизодов: {episodes_done} | eps: {epsilon:.4f}")

def choose_action(state, current_epsilon):
    if config.TRAIN_MODE and random.random() < current_epsilon:
        return random.choice(config.ACTIONS)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax().item())

def optimize_model():
    if len(memory) < config.BATCH_SIZE: return
    
    batch = random.sample(memory, config.BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    
    state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
    done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
    expected_q_values = reward_batch + (config.GAMMA * next_q_values * (1 - done_batch))

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target_network():
    target_net.load_state_dict(policy_net.state_dict())