import os
import glob
import shutil

def reset_training():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    target_files = [
        os.path.join(base_dir, "q_table_v2.npy"),
        os.path.join(base_dir, "state_v2.json")
    ]
    
    deleted_count = 0
    
    print("=== ЗАПУСК ПОЛНОГО СБРОСА ОБУЧЕНИЯ ===")
    
    # Удаляем файлы модели и состояния
    for file_path in target_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[УДАЛЕНО] {os.path.basename(file_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"[ОШИБКА] Не удалось удалить {os.path.basename(file_path)}: {e}")
                
    # Удаляем папку stats целиком
    stats_dir = os.path.join(base_dir, "stats")
    if os.path.exists(stats_dir):
        try:
            shutil.rmtree(stats_dir)
            print("[УДАЛЕНО] Папка 'stats' со всей статистикой")
            deleted_count += 1
        except Exception as e:
            print(f"[ОШИБКА] Не удалось удалить папку 'stats': {e}")

    if deleted_count == 0:
        print("Нечего удалять. Файлы обучения и статистики не найдены.")
    else:
        print(f"=== ГОТОВО! Удалено объектов: {deleted_count} ===")
        print("Теперь можно запускать main.py, обучение начнется с чистого листа.")

if __name__ == "__main__":
    reset_training()