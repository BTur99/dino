# heatmap.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_heatmap():
    model_file = "q_table_v2.npy"
    if not os.path.exists(model_file):
        print("Файл q_table_v2.npy не найден. Агенту нужно сначала поучиться!")
        return

    # Загружаем мозг
    q_table = np.load(model_file, allow_pickle=True).item()
    
    # Настройка шрифтов
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Tahoma', 'DejaVu Sans']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Тепловая карта решений (0=Бег, 1=Прыжок, 2=Присед)', fontsize=16, fontweight='bold')

    # --- ГРАФИК 1: Реакция на ПТИЦ ---
    # Оси: X - Расстояние (dist_bin от 0 до 4), Y - Высота птицы (height_bin от 0 до 2)
    # Фильтруем: bird_bin == 1, dino_height_bin == 0 (динозавр на земле)
    matrix_birds = np.full((3, 5), -1) # -1 значит "нет данных"
    
    for state, q_values in q_table.items():
        dist, height, bird, width, second, dino_y = state
        if bird == 1 and dino_y == 0 and width == 0 and second == 0: # Берем средние условия
            best_action = int(np.argmax(q_values))
            matrix_birds[height, dist] = best_action

    sns.heatmap(matrix_birds, annot=True, cmap="coolwarm", cbar=False, 
                xticklabels=["Пусто", "Далеко", "Средне", "Близко", "Критично"], 
                yticklabels=["Высоко", "Средне", "Низко"], ax=ax1)
    ax1.set_title("Реакция на Птиц", fontsize=14)
    ax1.set_xlabel("Расстояние до препятствия")
    ax1.set_ylabel("Высота птицы")

    # --- ГРАФИК 2: Реакция на КАКТУСЫ ---
    # Фильтруем: bird_bin == 0
    matrix_cactus = np.full((3, 5), -1)
    
    for state, q_values in q_table.items():
        dist, height, bird, width, second, dino_y = state
        if bird == 0 and dino_y == 0 and height == 0 and second == 0:
            best_action = int(np.argmax(q_values))
            matrix_cactus[width, dist] = best_action # Y-ось теперь ширина (0=узкий, 1=широкий)

    sns.heatmap(matrix_cactus[:2, :], annot=True, cmap="coolwarm", cbar=False, 
                xticklabels=["Пусто", "Далеко", "Средне", "Близко", "Критично"], 
                yticklabels=["Узкий", "Широкий"], ax=ax2)
    ax2.set_title("Реакция на Кактусы", fontsize=14)
    ax2.set_xlabel("Расстояние до препятствия")
    ax2.set_ylabel("Ширина кактуса")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_heatmap()