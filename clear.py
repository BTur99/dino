# reset.py
import os
import glob
import shutil

def reset_training():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    target_files = [
        os.path.join(base_dir, "q_table_v2.npy"),
        os.path.join(base_dir, "state_v2.json"),
        os.path.join(base_dir, "app.log") # НОВОЕ: Удаляем лог-файл
    ]
    
    deleted_count = 0
    
    print("=== ЗАПУСК ПОЛНОГО СБРОСА ===")
    
    for file_path in target_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[УДАЛЕНО] {os.path.basename(file_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"[ОШИБКА] Не удалось удалить {os.path.basename(file_path)}: {e}")
                
    stats_dir = os.path.join(base_dir, "stats")
    if os.path.exists(stats_dir):
        try:
            shutil.rmtree(stats_dir)
            print("[УДАЛЕНО] Папка 'stats' со всей статистикой")
            deleted_count += 1
        except Exception as e:
            print(f"[ОШИБКА] Не удалось удалить папку 'stats': {e}")

    chrome_profiles = glob.glob(os.path.join(base_dir, "chrome_profile_*"))
    for folder in chrome_profiles:
        try:
            shutil.rmtree(folder, ignore_errors=True)
            print(f"[УДАЛЕНО] {os.path.basename(folder)}")
            deleted_count += 1
        except Exception as e:
            print(f"[ОШИБКА] Не удалось удалить {folder}: {e}")

    if deleted_count == 0:
        print("Нечего удалять. Всё и так чисто.")
    else:
        print(f"=== ГОТОВО! Удалено объектов: {deleted_count} ===")
        print("Теперь можно запускать main.py, всё начнется с абсолютно чистого листа.")

if __name__ == "__main__":
    reset_training()