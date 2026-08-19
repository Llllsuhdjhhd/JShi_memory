import sqlite3
import json
import os
from pathlib import Path

def inspect_latest_sim():
    # 1. 找到最新的模拟目录 (在新的 outputs/sim_runs 架构下)
    base_dir = Path(__file__).parent / "outputs" / "sim_runs"
    sim_dirs = sorted(
        list(base_dir.glob("sim_run_*")) + list(base_dir.glob("sim_5ch_*")),
        key=os.path.getmtime,
        reverse=True,
    )
    
    if not sim_dirs:
        # 兼容性回退：检查旧的 tests 根目录（如果还没清理）
        old_base = Path(__file__).parents[2]
        sim_dirs = sorted(old_base.glob("hongloumeng_sim_*"), key=os.path.getmtime, reverse=True)
        
    if not sim_dirs:
        print("未找到任何模拟运行目录。")
        return
    
    latest_dir = sim_dirs[0]
    db_path = latest_dir / "rems_sim.db"
    
    print(f"\n{'='*80}")
    print(f" 正在检查最新模拟结果: {latest_dir}")
    print(f"{'='*80}")
    
    if not db_path.exists():
        print(f"数据库文件不存在: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. 查看封存事件 (Events)
    print(f"\n[1. 封存事件 (Sealed Events)]")
    try:
        # 正确列名：create_time, summaries, role_list
        cursor.execute("SELECT event_id, summaries, role_list FROM events ORDER BY create_time")
        rows = cursor.fetchall()
        for eid, summaries_json, role_list_json in rows:
            summaries = json.loads(summaries_json) if summaries_json else {}
            role_entries = json.loads(role_list_json) if role_list_json else []
            
            # 手动计算情感能量 (示例中取最大值或总和，这里简化展示)
            l1_summary = summaries.get('L1', '无摘要')
            print(f"  - 事件ID: {eid}")
            print(f"    摘要 (L1): {l1_summary[:100]}...")
            if role_entries:
                # 白皮书 §2.5 已用 8 维 BasicEmotionVector 取代旧 vedana(joy/suffering) 二维结构。
                # 这里展示 anger/joy/sadness/trust 4 个最常用维度，以及合成后的 valence/arousal。
                first_role = role_entries[0]
                em = first_role.get('emotional_model', {}) or {}
                emo = em.get('emotion', {}) or {}
                anger = emo.get('anger', 0.0) or 0.0
                joy = emo.get('joy', 0.0) or 0.0
                sadness = emo.get('sadness', 0.0) or 0.0
                trust = emo.get('trust', 0.0) or 0.0
                valence = em.get('valence', 0.0) or 0.0
                arousal = em.get('arousal', 0.0) or 0.0
                print(
                    "    情感状态 (首位角色): "
                    f"怒={anger:.2f}, 喜={joy:.2f}, 哀={sadness:.2f}, 信={trust:.2f} | "
                    f"valence={valence:+.2f}, arousal={arousal:.2f}"
                )
            print("-" * 40)
    except Exception as e:
        print(f"查询事件失败: {e}")

    # 3. 查看角色画像 (Semantic Cards)
    print(f"\n[2. 角色语义卡片 (Semantic Cards)]")
    try:
        cursor.execute("SELECT role_id, data FROM semantic_cards")
        rows = cursor.fetchall()
        for rid, data_json in rows:
            data = json.loads(data_json) if data_json else {}
            print(f"  ■ 角色ID: {rid}")
            # 注意：实际 data 结构依具体实现而定，通常包含 identity, card_data 等
            print(f"    核心身份: {data.get('identity', '未知')}")
            keys = data.get('semantic_card', {}).get('keys', {})
            if keys:
                print(f"    语义词云: {', '.join(list(keys.keys())[:10])}")
            print("-" * 60)
    except Exception as e:
        print(f"查询语义卡片失败: {e}")

    # 4. 查看白描记录 (White-Painting)
    print(f"\n[3. 白描演进时间线 (White-Painting)]")
    try:
        cursor.execute("SELECT role_id, event_id, role_summary, create_time FROM white_painting_entries ORDER BY create_time")
        rows = cursor.fetchall()
        for rid, eid, summary, ctime in rows:
            print(f"  [{ctime}] 角色 {rid} 在事件 {eid} 中:")
            print(f"    > {summary}")
    except Exception as e:
        print(f"查询白描记录失败: {e}")

    conn.close()

if __name__ == "__main__":
    inspect_latest_sim()
