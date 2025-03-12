import openpyxl
from openpyxl.drawing.image import Image
import os

# 打开 Excel 文件
file_path = 'data/eval/表情+文本测试集.xlsx'
output_dir = 'data/eval/表情/extracted_images'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载工作簿
wb = openpyxl.load_workbook(file_path, data_only=True)

# 遍历所有工作表
for sheet_name in wb.sheetnames:
    sheet = wb[sheet_name]
    print(f"Processing sheet: {sheet_name}")

    # 遍历单元格，寻找图片
    for image in sheet._images:
        # 获取图片的位置信息
        anchor = image.anchor  # 例如 "A1"
        print(f"Image found at {anchor}")

        # 保存图片
        image_path = os.path.join(output_dir, f"{sheet_name}_{anchor}.png")
        image.image.save(image_path)
        print(f"Saved image to {image_path}")


# import pandas as pd

# # 假设数据文件路径为 input.xlsx，包含 session_id 和其他列
# input_file = 'data/eval/表情+文本测试集.xlsx'
# output_file = 'data/eval/表情+文本测试集-1.xlsx'

# # 读取 Excel 文件
# df = pd.read_excel(input_file)

# # 确保数据包含 session_id 列
# if 'session_id' not in df.columns:
#     raise ValueError("输入数据中缺少 session_id 列")

# # 按 session_id 分组，并对每组的 round 列生成递增序列
# df['round'] = df.groupby('session_id').cumcount() + 1

# # 将结果保存回新的 Excel 文件
# df.to_excel(output_file, index=False)

# print(f"已处理完成，结果保存至 {output_file}")


# import pandas as pd
# import os

# # 假设数据来自 Excel 文件
# input_file = 'data/eval/表情+文本测试集.xlsx'
# df = pd.read_excel(input_file)

# def is_image(content):
#     """
#     判断数据是否为图片。
#     """
#     if isinstance(content, str):
#         # 检查是否为路径或 URL
#         if content.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             return True
#         if content.lower().startswith(('http://', 'https://')) and content.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             return True
#     return False

# # 添加新列，标记 content 是图片还是文本
# df['is_image'] = df['content'].apply(is_image)

# # 将结果保存到新的 Excel 文件
# output_file = '/mnt/data/output_with_type.xlsx'
# df.to_excel(output_file, index=False)

# print(f"处理完成，结果保存到 {output_file}")
