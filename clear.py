# reset.py
import os
import glob
import shutil

def reset_training():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_files = [
        os.path.join(base_dir, "dino_dqn.pth"), # Обновили имя файла
        os.path.join(base_dir, "state_v2.json"),
        os.path.join(base_dir, "app.log")
    ]
    deleted_count = 0
    print("=== СБРОС ===")
    
    for file_path in target_files:
        if os.path.exists(file_path):
            try: os.remove(file_path); deleted_count += 1; print(f"[УДАЛЕНО] {os.path.basename(file_path)}")
            except Exception as e: print(f"[ОШИБКА] {e}")
                
    stats_dir = os.path.join(base_dir, "stats")
    if os.path.exists(stats_dir):
        try: shutil.rmtree(stats_dir); deleted_count += 1; print("[УДАЛЕНО] Папка 'stats'")
        except Exception as e: print(f"[ОШИБКА] {e}")

    chrome_profiles = glob.glob(os.path.join(base_dir, "chrome_profile_*"))
    for folder in chrome_profiles:
        try: shutil.rmtree(folder, ignore_errors=True); deleted_count += 1; print(f"[УДАЛЕНО] {os.path.basename(folder)}")
        except Exception as e: print(f"[ОШИБКА] {e}")

    print(f"=== ГОТОВО! Удалено: {deleted_count} ===")

if __name__ == "__main__":
    reset_training()