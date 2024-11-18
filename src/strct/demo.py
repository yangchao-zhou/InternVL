'''
获取测试集中的结构化信息
'''
import os
import requests
import pandas as pd

# fetch_npc_profile 函数
def fetch_npc_profile(cid: str):
    path = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(path, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{cid}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            npc_profile = f.read()
        return npc_profile

    url = f"https://llm-push-chat.linke.ai/npc/struct_v2?cid={cid}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    try:
        return_data = data["data"]["npc_struct_v2_list"][0]["npc_profile"]
    except Exception as e:
        return None
    return return_data

# 读取 Excel 文件
def read_excel(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# 根据 session_id 去重并获取数据
def process_npc_data_unique(df, id_column, output_fields):
    # 根据 session_id 去重
    df_unique = df.drop_duplicates(subset=[id_column])

    processed_data = []
    for _, row in df_unique.iterrows():
        session_id = row[id_column]
        line = session_id.split("-")
        cid_1 = line[0]
        cid_2 = line[1]
        if len(cid_1)<10:
            cid = cid_1
        else:
            cid = cid_2

        # 调用 fetch_npc_profile 获取结构化信息
        profile_data = fetch_npc_profile(cid)

        processed_data.append({
            "用途": row["type"],
            "msg_id": row["session_id"],
            "角色名": row["NPC_name"],
            "测试环境sid": row["角色sid"],
            "intro": row["intro"],
            "greeting": row["greeting"],
            "struct_info": profile_data
        })

    return pd.DataFrame(processed_data, columns=output_fields)

# 保存为新的 Excel 文件
def save_to_excel(df, output_file):
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    # Excel 文件路径
    input_file = "/mnt/workspace/yangchao.zhou/opt/InternVL/data/eval/表情+文本测试集.xlsx"
    output_file = "/mnt/workspace/yangchao.zhou/opt/InternVL/data/eval/结构化信息.xlsx"

    # 读取 Excel 文件
    df = read_excel(input_file, sheet_name="固定话术-跑模型以此为准！")

    # 处理数据，去重并生成指定字段
    output_fields = ["用途", "msg_id", "角色名", "测试环境sid", "intro", "greeting", "struct_info"]
    processed_df = process_npc_data_unique(df, id_column="session_id", output_fields=output_fields)

    # 保存结果到新的 Excel 文件
    save_to_excel(processed_df, output_file)

    print("处理完成！")
