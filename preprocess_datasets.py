import json
import os
import pandas as pd

# Paths to your datasets
datasets = {
    "IPC": "jsons/ipc.json",
    "CrPC": "jsons/crpc.json",
    "NIA": "jsons/nia.json"
}

processed_data = []

for act_name, path in datasets.items():
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Ensure each dataset has these keys
        section_no = item.get("section", "")
        title = item.get("section_title", "")
        content = item.get("section_desc", "")

        processed_data.append({
            "act": act_name.lower(),
            "chapter": item.get("chapter", "Unknown"),
            "section_no": section_no,
            "title": title,
            "content": content,
            "version": "2025.10"
        })

# Convert to DataFrame
df = pd.DataFrame(processed_data)

# Save cleaned CSV for reference
df.to_csv("legal_data_cleaned.csv", index=False)
print("Datasets preprocessed and merged successfully!")
