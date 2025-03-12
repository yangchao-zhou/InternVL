import os
from PIL import Image
from time import time

def get_all_file_paths(directory, extensions=('.jpg', '.jpeg', '.png', '.gif')):
    """获取指定目录下的所有图片文件路径"""
    image_paths = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def resize_image(image_path, output_path, max_width=200, max_height=200):
    """压缩图片并保存"""
    img = Image.open(image_path)
    img.thumbnail((max_width, max_height))  # 保持纵横比，缩小到最大宽度和高度
    img.save(output_path, quality=50)  # 保存压缩后的图片
    return output_path  # 返回压缩后保存的路径

def main():
    """
    主函数，执行图片压缩任务
    """
    images_dir = 'data/pic/测试图片'  # 输入目录
    output_dir = 'data/pic/测试图片_缩略/'  # 输出目录

    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图片文件路径
    images_path = get_all_file_paths(images_dir)

    # 用于保存结果
    results = []

    for image_path in images_path:
        # 构造输出图片路径
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)

        # 执行图片压缩
        start_time = time()
        result = resize_image(image_path, output_path)
        end_time = time()

        # 打印处理时间
        print(f"{image_path} took {end_time - start_time:.4f} seconds")
        



if __name__ == "__main__":
    main()
