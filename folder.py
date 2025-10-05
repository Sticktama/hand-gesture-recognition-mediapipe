import os
import pandas as pd

# Paths
base_dir = r"C:\\Users\Dell\\Downloads\\FSL-105 A dataset for recognizing 105 Filipino sign language videos\\FSL-105 A dataset for recognizing 105 Filipino sign language videos\\clips\\clips"   # Replace with the folder where 0-105 are located
csv_file = r"C:\\Users\Dell\\Downloads\\FSL-105 A dataset for recognizing 105 Filipino sign language videos\\FSL-105 A dataset for recognizing 105 Filipino sign language videos\\labels.csv"  # Replace with your CSV file path

# Load CSV
df = pd.read_csv(csv_file)

# Create dictionary mapping id -> label
id_to_label = dict(zip(df["id"], df["label"]))

# Rename folders
for folder_id, label in id_to_label.items():
    old_path = os.path.join(base_dir, str(folder_id))
    new_path = os.path.join(base_dir, label)

    if os.path.exists(old_path):
        # Ensure no overwriting happens if folder already exists
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
        else:
            print(f"Skipped (already exists): {new_path}")
    else:
        print(f"Folder not found: {old_path}")
