from datasets import load_dataset
from collections import Counter

PARQUET_PATH = r"C:\Users\pendr\Desktop\CapNav\dataset_local\capnav_v0_no_answer.parquet"

ds = load_dataset("parquet", data_files=PARQUET_PATH)["train"]

print("rows:", len(ds))
print("unique scenes:", len(set(ds["scene_id"])))
print("unique agents:", len(set(ds["agent_name"])))
print("columns:", ds.column_names)

c = Counter(ds["scene_id"])
print("sample scene counts:", c.most_common(5))
