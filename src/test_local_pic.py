import os
from openai import OpenAI
import jinja2
import pandas as pd
import xlsxwriter
from PIL import Image
from time import time

# Initialize the OpenAI client
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')

# Get the model ID
model_name = client.models.list().data[0].id

prompt_path = "prompt/dis_pic_yc_1.tmpl"

cell_height = 200

# 加载模板
def load_prompt_template(template_file):
    with open(template_file, 'r') as f:
        template = f.read()
    return jinja2.Template(template)

template = load_prompt_template(prompt_path)
prompt = template.render({})

def get_all_file_paths(directory):
    """
    获取指定目录下的所有文件路径。
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

def get_result(image_path):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': prompt,
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': image_path,
                },
            }],
        }],
        temperature=0.8,
        top_p=0.8
    )
    return response.choices[0].message.content

def get_image_dimensions(image_path):
    """
    获取图片的宽度和高度。
    """
    with Image.open(image_path) as img:
        return img.size  # 返回宽度和高度 (width, height)

def main():
    """
    主函数，用于处理指定目录下的所有图像文件，并将结果写入Excel文件。
    """
    images_dir = '/mnt/workspace/yangchao.zhou/opt/InternVL/data/pic/测试图片'
    output_excel = 'output/output.xlsx'

    images_path = get_all_file_paths(images_dir)
    results = []

    total_time = 0  # 累加所有图片处理时间

    for image_path in images_path:
        start_time = time()
        result = get_result(image_path)
        end_time = time()
        
        # 计算每张图片的处理时间
        processing_time = end_time - start_time
        total_time += processing_time
        
        print(f"{image_path} took {processing_time:.4f} seconds")
        
        results.append({'Image Path': image_path, 'Result': result, 'Processing Time': processing_time})

    # 计算平均处理时间
    average_time = total_time / len(images_path) if images_path else 0
    print(f"Average processing time: {average_time:.4f} seconds per image")

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 将结果和图片写入Excel
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Results', index=False, startcol=2)  # 写入结果到Excel

        # 获取Excel的工作簿和工作表
        workbook = writer.book
        worksheet = writer.sheets['Results']

        # 设置每行的高度为50
        for row_num in range(len(df)):
            worksheet.set_row(row_num + 1, cell_height)  # 从第2行开始设置高度为50

        # 为了避免图片堆叠，动态计算图片大小
        for idx, image_path in enumerate(df['Image Path']):
            row = idx + 1  # 每张图片放在独立行
            col = 0  # 假设图片插入到第1列

            # 获取图片的原始尺寸
            image_width, image_height = get_image_dimensions(image_path)

            # 计算缩放比例：确保图片高度不超过单元格的高度
            y_scale = cell_height / image_height
            # 根据比例调整宽度
            x_scale = y_scale * (image_width / image_height)

            # 插入图片到单元格，调整大小
            worksheet.insert_image(row, col, image_path, {'x_scale': x_scale, 'y_scale': y_scale})

    print(f"Results have been written to {output_excel}")

if __name__ == "__main__":
    main()
