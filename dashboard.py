# web_dashboard.py
import streamlit as st
import json
import os
import glob
import pandas as pd
import time

# Настройка страницы (без иконки-эмодзи)
st.set_page_config(page_title="Dino AI Trainer", layout="wide")

st.title("Дашборд обучения динозавриков")
st.markdown("Автоматическое обновление каждые 3 секунды...")

def load_data():
    stats_dir = "stats"
    data = []
    
    for filepath in glob.glob(os.path.join(stats_dir, "stats_*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                name = os.path.basename(filepath).replace("stats_", "").replace(".json", "")
                stats = json.load(f)
                stats['name'] = name
                data.append(stats)
        except Exception:
            pass
            
    global_state = {}
    if os.path.exists("state_v2.json"):
        try:
            with open("state_v2.json", "r", encoding="utf-8") as f:
                global_state = json.load(f)
        except Exception:
            pass
            
    return data, global_state

placeholder = st.empty()

while True:
    try:
        data, global_state = load_data()
        
        with placeholder.container():
            # --- ВЕРХНЯЯ ПАНЕЛЬ ---
            col1, col2, col3, col4 = st.columns(4)
            
            episodes = global_state.get("episodes_done", 0)
            eps = global_state.get("epsilon", 0)
            alpha = global_state.get("current_alpha", 0)
            
            col1.metric("Сыграно эпизодов", episodes)
            col2.metric("Epsilon (Любопытство)", f"{eps:.4f}")
            col3.metric("Alpha (Скорость обучения)", f"{alpha:.4f}")
            
            if data:
                champ = max(data, key=lambda x: x.get("max_score", 0))
                col4.metric("Абсолютный чемпион", champ['name'], f"Рекорд: {champ['max_score']} очков")
            else:
                col4.metric("Абсолютный чемпион", "Нет данных")
            
            st.divider()

            # --- ОСНОВНЫЕ ГРАФИКИ ---
            if data:
                df = pd.DataFrame(data)
                df = df.sort_values(by="avg_last_100", ascending=False)
                
                col_chart, col_history = st.columns([1, 1])
                
                with col_chart:
                    st.subheader("Топ по среднему результату (100 игр)")
                    chart_data = df.set_index('name')['avg_last_100']
                    st.bar_chart(chart_data)
                    
                with col_history:
                    st.subheader("Динамика последних 100 игр (Топ-3)")
                    top_3 = df.head(3)
                    
                    history_dict = {}
                    for _, row in top_3.iterrows():
                        history = row.get("last_100_scores", [])
                        if history:
                            # ИСПРАВЛЕНИЕ ОШИБКИ: Выравниваем длину массивов
                            # Если у агента меньше 100 игр, дополняем начало пустотами (None)
                            if len(history) < 100:
                                padding = [None] * (100 - len(history))
                                history_dict[row['name']] = padding + history
                            else:
                                history_dict[row['name']] = history[-100:]
                                
                    if history_dict:
                        st.line_chart(pd.DataFrame(history_dict))
                    else:
                        st.info("Игроки еще не сыграли достаточно игр.")
            else:
                st.warning("Ожидание данных от агентов... Убедись, что main.py запущен.")

    except Exception as e:
        st.error(f"Ошибка чтения данных: {e}")

    time.sleep(3)