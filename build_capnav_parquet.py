import os
import re
import json
from typing import Any, Dict, List, Optional

# =========================
# 0) 配置区：按你的本机路径修改
# =========================

QA_DIR = r"C:\Users\pendr\Desktop\CapNav\ground_truth\qa"              # 存放 *_QA.json
GRAPH_DIR = r"C:\Users\pendr\Desktop\CapNav\ground_truth\graphs"      # 存放 *-graph.json（若与 QA 同目录可填 QA_DIR）
AGENT_PROFILE_PATH = r"C:\Users\pendr\Desktop\CapNav\agent_profile.json"

OUT_DIR = r"C:\Users\pendr\Desktop\CapNav\dataset_local"
OUT_PARQUET = "capnav_v0_no_answer.parquet"

# question_id 的生成方式：
# - "global": 全局递增 00000001, 00000002, ...
# - "per_scene": 每个 scene 内递增 HM3D00000_0001, HM3D00000_0002, ...
QUESTION_ID_MODE = "global"  # or "per_scene"

# agent 顺序是否排序（建议 True，避免不同机器输出顺序不同）
SORT_AGENT_NAMES = True

# =========================
# 1) 工具函数
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_scene_id_from_qa_filename(filename: str) -> Optional[str]:
    # 期望：HM3D00000_QA.json
    m = re.match(r"^(?P<scene_id>.+)_QA\.json$", filename)
    return m.group("scene_id") if m else None

def graph_filename_for_scene(scene_id: str) -> str:
    # 期望：HM3D00000-graph.json
    return f"{scene_id}-graph.json"

def locate_graph_path(scene_id: str) -> Optional[str]:
    graph_fn = graph_filename_for_scene(scene_id)
    p1 = os.path.join(GRAPH_DIR, graph_fn)
    if os.path.exists(p1):
        return p1
    # 兜底：graph 与 QA 同目录
    p2 = os.path.join(QA_DIR, graph_fn)
    if os.path.exists(p2):
        return p2
    return None

def build_scene_nodes(graph_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    只保留节点 id 和 name，不包含 description / position / edges
    输出：[{ "node_id": "...", "name": "..." }, ...]
    """
    nodes = graph_json.get("nodes", [])
    out: List[Dict[str, str]] = []
    for n in nodes:
        node_id = n.get("id")
        name = n.get("name")
        if node_id is None or name is None:
            continue
        out.append({"node_id": str(node_id), "name": str(name)})
    return out

def list_qa_files(qa_dir: str) -> List[str]:
    fns = [fn for fn in os.listdir(qa_dir) if fn.endswith("_QA.json")]
    fns.sort()
    return fns

# =========================
# 2) 主逻辑：构建 rows
# =========================

def build_rows() -> List[Dict[str, Any]]:
    agents = read_json(AGENT_PROFILE_PATH)
    agent_names = [a.get("agent_name") for a in agents if isinstance(a, dict) and a.get("agent_name")]
    if not agent_names:
        raise ValueError("agent_profile.json 中未找到任何 agent_name。")

    if SORT_AGENT_NAMES:
        agent_names = sorted(agent_names)

    qa_files = list_qa_files(QA_DIR)
    if not qa_files:
        raise FileNotFoundError(f"在目录中未找到任何 *_QA.json：{QA_DIR}")

    rows: List[Dict[str, Any]] = []
    global_qid = 0

    for qa_fn in qa_files:
        scene_id = parse_scene_id_from_qa_filename(qa_fn)
        if scene_id is None:
            print(f"[WARN] 无法解析 scene_id，跳过：{qa_fn}")
            continue

        qa_path = os.path.join(QA_DIR, qa_fn)
        graph_path = locate_graph_path(scene_id)
        if graph_path is None:
            print(f"[WARN] 找不到 graph 文件（{graph_filename_for_scene(scene_id)}），跳过 scene：{scene_id}")
            continue

        qa_list = read_json(qa_path)
        if not isinstance(qa_list, list):
            print(f"[WARN] QA 文件不是 list，跳过：{qa_path}")
            continue

        graph_json = read_json(graph_path)
        scene_nodes = build_scene_nodes(graph_json)

        per_scene_qid = 0
        added = 0
        skipped = 0

        for qa in qa_list:
            if not isinstance(qa, dict):
                skipped += 1
                continue

            question = qa.get("question")
            scene_type = qa.get("scene_type")
            if question is None or scene_type is None:
                skipped += 1
                continue

            for agent_name in agent_names:
                if QUESTION_ID_MODE == "global":
                    global_qid += 1
                    question_id = f"{global_qid:08d}"
                elif QUESTION_ID_MODE == "per_scene":
                    per_scene_qid += 1
                    question_id = f"{scene_id}_{per_scene_qid:04d}"
                else:
                    raise ValueError("QUESTION_ID_MODE 只能是 'global' 或 'per_scene'")

                rows.append({
                    "question_id": question_id,
                    "question": question,
                    "scene_id": scene_id,
                    "scene_type": scene_type,
                    "scene_nodes": scene_nodes,   # nested list[struct]
                    "agent_name": agent_name,
                    "answer": None                # 未来回填 ground truth
                })
                added += 1

        print(f"[OK] scene={scene_id} | qa={len(qa_list)} | agents={len(agent_names)} | rows_added={added} | skipped_qa_items={skipped}")

    print(f"[DONE] total_rows={len(rows)}")
    return rows

# =========================
# 3) 保存 Parquet
# =========================

def save_parquet(rows: List[Dict[str, Any]], out_path: str) -> None:
    """
    使用 pandas + pyarrow 写 parquet，支持 nested list[struct] 的 scene_nodes
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

# =========================
# 4) 校验：用 datasets 读取 parquet（推荐、最稳定）
# =========================

def quick_validate_parquet(out_path: str) -> None:
    """
    用 Hugging Face datasets 的 parquet reader 校验 nested 结构是否正确
    """
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=out_path)
    sample = ds["train"][0]

    v = sample.get("scene_nodes")
    ok = isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict))
    if not ok:
        raise ValueError(
            "scene_nodes 结构不正确。"
            f"\n实际类型：{type(v)}\n样例值前 200 字符：{str(v)[:200]}"
        )

    print("[VALID] parquet verified with datasets.load_dataset()")
    print("[VALID] columns:", ds["train"].column_names)
    print("[VALID] scene_nodes sample:", v[:3])

# =========================
# 5) 入口
# =========================

def main():
    ensure_dir(OUT_DIR)
    rows = build_rows()

    if not rows:
        raise RuntimeError("rows 为空：没有生成任何数据。请检查 GRAPH_DIR/QA_DIR 是否匹配、graph 文件命名是否为 <scene_id>-graph.json。")

    out_path = os.path.join(OUT_DIR, OUT_PARQUET)
    save_parquet(rows, out_path)
    print(f"[WRITE] {out_path}")

    quick_validate_parquet(out_path)

if __name__ == "__main__":
    main()
