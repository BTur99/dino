import os
import json
import glob
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Tahoma', 'DejaVu Sans']

def load_stats():
    data = {}
    # Ищем файлы в папке stats
    for filepath in glob.glob(os.path.join("stats", "stats_*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Берем только имя файла (убираем путь stats/)
                filename = os.path.basename(filepath)
                name = filename.replace("stats_", "").replace(".json", "")
                data[name] = json.load(f)
        except Exception:
            pass 
    return data

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('ТОП Динозавриков')

print("Дашборд запущен! Ожидание данных из папки 'stats'...")

while True:
    try:
        data = load_stats()
        if not data:
            plt.pause(2)
            continue

        ax.clear()
        
        sorted_data = sorted(data.items(), key=lambda x: x[1].get("avg_last_100", 0), reverse=True)
        
        names = [item[0] for item in sorted_data]
        averages = [item[1].get("avg_last_100", 0) for item in sorted_data]
        
        absolute_champ_name = max(data.items(), key=lambda x: x[1].get("max_score", 0))[0]
        absolute_champ_max = data[absolute_champ_name].get("max_score", 0)

        colors = []
        for i in range(len(names)):
            if i == 0: colors.append("#FFD700") 
            elif i == 1: colors.append("#C0C0C0") 
            elif i == 2: colors.append("#CD7F32") 
            else: colors.append("#4C72B0") 

        bars = ax.bar(names, averages, color=colors, edgecolor="black")
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
            
        ax.set_title("Средний результат за 100 игр", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Очки", fontsize=12)
        
        max_val = max(averages) if averages else 100
        ax.set_ylim(0, max_val * 1.2)
        
        ax.text(0.98, 0.95, 
                f"Абсолютный рекорд у {absolute_champ_name}\nРекорд: {absolute_champ_max} очков", 
                transform=ax.transAxes, 
                fontsize=12, fontweight='bold',
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gold', boxstyle='round,pad=0.5'))
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.pause(2)
        
    except KeyboardInterrupt:
        print("Дашборд закрыт.")
        break