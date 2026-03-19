import torch
import os
import zipfile
import shutil

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "best_hybrid_model")
REPACKED_PATH = os.path.join(CURRENT_DIR, "repacked_model.pt")

print(f"Model Directory: {MODEL_DIR}")

def zip_directory(folder_path, output_path):
    print(f"Zipping {folder_path} to {output_path}...")
    # Use ZIP_STORED (no compression) which is standard for torch serialization
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Exclude dot-directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                # Prepend 'archive/' to the relative path
                rel_path = os.path.relpath(file_path, folder_path)
                arcname = os.path.join('archive', rel_path)
                # Ensure forward slashes for zip compatibility?
                arcname = arcname.replace(os.path.sep, '/')
                
                zipf.write(file_path, arcname)
    print("Zipping complete.")

if os.path.exists(MODEL_DIR):
    try:
        if os.path.exists(REPACKED_PATH):
            os.remove(REPACKED_PATH)
            
        zip_directory(MODEL_DIR, REPACKED_PATH)
        
        # Verify zip contents
        print("Verifying zip contents (first 10):")
        with zipfile.ZipFile(REPACKED_PATH, 'r') as z:
            print(z.namelist()[:10])
            if 'archive/data.pkl' not in z.namelist():
                print("CRITICAL: archive/data.pkl not found in zip!")
            
        
        print("Attempting to load repacked model...")
        device = torch.device("cpu")
        state_dict = torch.load(REPACKED_PATH, map_location=device)
        print("SUCCESS: Model loaded from repacked file!")
        print(f"Keys: {list(state_dict.keys())[:5]}")
    except Exception as e:
        print(f"FAILED to load repacked model: {e}")
        # import traceback
        # traceback.print_exc()
else:
    print(f"Directory {MODEL_DIR} does not exist.")
