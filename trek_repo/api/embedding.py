import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

input_dir = r"C:\Users\qijin\OneDrive\Desktop\2025-CUHK-PHD\Benchmark\api\data\hotel_data"
output_dir = r"C:\Users\qijin\OneDrive\Desktop\2025-CUHK-PHD\Benchmark\api\data\hotel_data\amenities_group_embedding"
os.makedirs(output_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
total = len(csv_files)

for idx, fname in enumerate(csv_files):
    city = fname.replace('_hotel.csv', '')
    df = pd.read_csv(os.path.join(input_dir, fname))
    # 注意这里是 amenities_group
    group_strs = [", ".join(eval(x)) for x in df["amenities_group"]]
    embeds = model.encode(group_strs, prompt_name="query")
    np.save(os.path.join(output_dir, f"{city}_amenities_group.npy"), embeds)
    print(f"Processing {city} ({idx+1}/{total})")  # 进度打印

print("全部城市 embedding 已完成！")
