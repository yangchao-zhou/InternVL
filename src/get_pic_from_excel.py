import os
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from PIL import Image as PILImage

# 读取 Excel 文件并处理图片
def save_images_from_excel(excel_path, sheet_name, output_dir):
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 载入 Excel 文件
    workbook = load_workbook(excel_path)
    sheet = workbook[sheet_name]

    image_count = 0
    
    # 遍历所有图片
    for img in sheet._images:
        image_count += 1
        
        try:
            # 获取图片的数据，调用 _data 方法获取字节流
            img_data = img._data()  # 需要调用 _data 方法
            
            # 将图片数据保存到文件
            image_filename = os.path.join(output_dir, f"image_{image_count}.png")
            
            # 使用 Pillow 保存图片
            pil_image = PILImage.open(BytesIO(img_data))
            pil_image.save(image_filename)
            print(f"Saved {image_filename}")
        
        except Exception as e:
            # 如果发生异常，输出错误信息，继续处理下一张图片
            print(f"Error processing image {image_count}: {e}")
            continue

# 示例：使用路径
input_excel_path = "data/eval/表情+文本测试集.xlsx"
output_image_dir = "data/eval/表情/extracted_images/"
save_images_from_excel(input_excel_path, sheet_name="固定话术-跑模型以此为准！", output_dir=output_image_dir)
