import requests
import json
import re
from PIL import Image, ImageDraw
import os
import time

from openai import OpenAI

# 初始化模型客户端
doubao_client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="a44785a4-cc83-4097-8906-4df873639e5c",
    default_headers={"ark-beta-image-process": "true"}
)

qwen_client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-c566575421a14a968638b7f47b2d57da"
)

# 新增doubao-seed-1.8模型客户端
doubao_client_18 = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="a477907a-7541-46d4-856a-44963a8dd203",
    default_headers={"ark-beta-image-process": "true"}
)

# 读取提示词
script_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(script_dir, 'prompt.pd')
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read().strip()

# 图片URL列表
image_urls = [
    "https://zhuoyu.tos-cn-beijing.volces.com/img1.jpg",
    "https://zhuoyu.tos-cn-beijing.volces.com/img2.jpg"
]

# 下载图片
def download_image(url, save_path="temp_image.jpg"):
    """下载图片到本地，失败时尝试使用本地同名文件"""
    # 从URL中提取图片名称
    image_name = os.path.basename(url)
    # 构建本地图片的完整路径
    local_image_path = os.path.join(script_dir, image_name)
    
    # 先检查本地是否有同名图片
    if os.path.exists(local_image_path):
        print(f"使用本地{image_name}")
        return local_image_path
    
    # 构建临时图片的完整路径
    temp_image_path = os.path.join(script_dir, save_path)
    
    # 尝试下载图片
    response = requests.get(url)
    if response.status_code == 200 and response.content:
        with open(temp_image_path, 'wb') as f:
            f.write(response.content)
        return temp_image_path
    else:
        # 下载失败，再次检查本地是否有同名图片
        if os.path.exists(local_image_path):
            print(f"图片下载失败，状态码：{response.status_code}，使用本地{image_name}")
            return local_image_path
        else:
            raise Exception(f"图片下载失败，状态码：{response.status_code}，本地也没有找到{image_name}")

# 提取bbox信息
def extract_bboxes(model_output):
    """从模型输出中提取原始bbox信息"""
    # 递归从JSON结构中提取bbox
    def extract_from_json(data):
        bboxes = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'bbox':
                    bboxes.append(value)
                else:
                    bboxes.extend(extract_from_json(value))
        elif isinstance(data, list):
            for item in data:
                bboxes.extend(extract_from_json(item))
        return bboxes
    
    original_bboxes = []
    try:
        # 尝试解析JSON
        if isinstance(model_output, str):
            parsed = json.loads(model_output)
        else:
            parsed = model_output
        original_bboxes = extract_from_json(parsed)
    except json.JSONDecodeError:
        # 如果不是JSON，尝试用正则表达式提取
        if isinstance(model_output, str):
            bbox_pattern = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
            matches = re.findall(bbox_pattern, model_output)
            for match in matches:
                try:
                    bbox = json.loads(match)
                    if len(bbox) == 4:
                        original_bboxes.append(bbox)
                except:
                    continue
    
    return original_bboxes

# 获取下一个序列编号
def get_next_sequence_number():
    """获取下一个可用的序列编号（0-100）"""
    for i in range(101):  # 从0到100
        img_path = os.path.join(script_dir, f"result_with_bboxes_{i}.jpg")
        bbox_path = os.path.join(script_dir, f"bbox_info_{i}.txt")
        if not os.path.exists(img_path) and not os.path.exists(bbox_path):
            return i
    return 0  # 如果所有编号都已使用，返回0

# 调整bbox坐标
def adjust_bboxes(bboxes, image_width, image_height):
    """根据图片尺寸调整bbox坐标
    1. 如果bbox坐标在0-1之间，认为是相对坐标，转换为绝对像素值
    2. 否则直接使用模型返回的坐标，确保为整数
    """
    adjusted_bboxes = []
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        
        # 检查是否为相对坐标（值在0-1之间）
        if max(x1, y1, x2, y2) <= 1.0:
            # 转换为绝对像素值
            x1 = int(x1 * image_width)
            y1 = int(y1 * image_height)
            x2 = int(x2 * image_width)
            y2 = int(y2 * image_height)
        else:
            # 直接使用模型返回的坐标，确保为整数
            x1 = int(x1*image_width/1000)
            y1 = int(y1*image_height/1000)
            x2 = int(x2*image_width/1000)
            y2 = int(y2*image_height/1000)
        
        # 确保坐标在图片范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)
        
        adjusted_bboxes.append([x1, y1, x2, y2])
    return adjusted_bboxes

# 绘制bbox
def draw_bboxes(image_path, doubao_bboxes, qwen_bboxes, doubao_bboxes_18, output_path):
    """在图片上绘制bboxes"""
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # 获取图片尺寸
        width, height = img.size
        
        # 调整bbox坐标
        adjusted_doubao_bboxes = adjust_bboxes(doubao_bboxes, width, height)
        adjusted_qwen_bboxes = adjust_bboxes(qwen_bboxes, width, height)
        adjusted_doubao_bboxes_18 = adjust_bboxes(doubao_bboxes_18, width, height)
        
        # 绘制豆包的bbox（绿色）
        for bbox in adjusted_doubao_bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # 绘制qwen的bbox（红色）
        for bbox in adjusted_qwen_bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # 绘制豆包1.8的bbox（蓝色）
        for bbox in adjusted_doubao_bboxes_18:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        
        img.save(output_path, "JPEG")
        
        return adjusted_doubao_bboxes, adjusted_qwen_bboxes, adjusted_doubao_bboxes_18

# 保存bbox信息到文件
def save_bbox_info(seq_num, doubao_bboxes, qwen_bboxes, doubao_bboxes_18):
    """保存bbox信息到文件"""
    bbox_path = os.path.join(script_dir, f"bbox_info_{seq_num}.txt")
    with open(bbox_path, 'w', encoding='utf-8') as f:
        # 写入豆包模型的bbox
        f.write("豆包模型bbox结果：\n")
        for bbox in doubao_bboxes:
            x1, y1, x2, y2 = bbox
            f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox>\n")
        
        # 写入Qwen模型的bbox
        f.write("Qwen模型bbox结果：\n")
        for bbox in qwen_bboxes:
            x1, y1, x2, y2 = bbox
            f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox>\n")
        
        # 写入豆包1.8模型的bbox
        f.write("豆包1.8模型bbox结果：\n")
        for bbox in doubao_bboxes_18:
            x1, y1, x2, y2 = bbox
            f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox>\n")
    return bbox_path

# 主函数
def main():
    # 遍历图片列表
    for i, image_url in enumerate(image_urls):
        print(f"\n=== 开始处理第 {i+1} 张图片: {image_url} ===")
        
        # 下载图片
        temp_image_path = download_image(image_url, f"temp_image_{i}.jpg")
        
        # 调用豆包模型
        print("正在调用豆包模型...")
        start_time = time.time()
        doubao_response = doubao_client.chat.completions.create(
            model="doubao-seed-1-6-vision-250815",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            extra_body={"reasoning": {"effort": "high"}},
            temperature=0.1,
            top_p=0.7
        )
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 输出token数和耗时
        if hasattr(doubao_response, 'usage'):
            prompt_tokens = getattr(doubao_response.usage, 'prompt_tokens', 'N/A')
            completion_tokens = getattr(doubao_response.usage, 'completion_tokens', 'N/A')
            total_tokens = getattr(doubao_response.usage, 'total_tokens', 'N/A')
            print(f"豆包模型消耗token数: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")
        print(f"豆包模型推理耗时: {inference_time:.2f} 秒")
        
        # 调用qwen模型
        print("正在调用Qwen模型...")
        start_time = time.time()
        qwen_response = qwen_client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            extra_body={"enable_thinking":True,
            "vl_high_resolution_images":True
            },
            temperature=0.1,
            top_p=0.7
        )
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 输出token数和耗时
        if hasattr(qwen_response, 'usage'):
            prompt_tokens = getattr(qwen_response.usage, 'prompt_tokens', 'N/A')
            completion_tokens = getattr(qwen_response.usage, 'completion_tokens', 'N/A')
            total_tokens = getattr(qwen_response.usage, 'total_tokens', 'N/A')
            print(f"Qwen模型消耗token数: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")
        print(f"Qwen模型推理耗时: {inference_time:.2f} 秒")
        
        # 调用豆包1.8模型
        print("正在调用豆包1.8模型...")
        start_time = time.time()
        doubao_response_18 = doubao_client_18.chat.completions.create(
            model="ep-20251217093342-w9jl6",  # 使用指定的模型ID
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            extra_body={"reasoning": {"effort": "high"}},
            temperature=0.1,
            top_p=0.7
        )
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 输出token数和耗时
        if hasattr(doubao_response_18, 'usage'):
            prompt_tokens = getattr(doubao_response_18.usage, 'prompt_tokens', 'N/A')
            completion_tokens = getattr(doubao_response_18.usage, 'completion_tokens', 'N/A')
            total_tokens = getattr(doubao_response_18.usage, 'total_tokens', 'N/A')
            print(f"豆包1.8模型消耗token数: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")
        print(f"豆包1.8模型推理耗时: {inference_time:.2f} 秒")
        
        # 提取bboxes
        print("正在提取bbox信息...")
        doubao_output = doubao_response.choices[0].message.content
        qwen_output = qwen_response.choices[0].message.content
        doubao_output_18 = doubao_response_18.choices[0].message.content
        
        doubao_bboxes = extract_bboxes(doubao_output)
        qwen_bboxes = extract_bboxes(qwen_output)
        doubao_bboxes_18 = extract_bboxes(doubao_output_18)
        
        # 获取序列编号
        seq_num = get_next_sequence_number()
        
        # 绘制并保存结果图片
        print("正在绘制结果图片...")
        output_image_path = os.path.join(script_dir, f"result_with_bboxes_{seq_num}.jpg")
        adjusted_doubao_bboxes, adjusted_qwen_bboxes, adjusted_doubao_bboxes_18 = draw_bboxes(
            temp_image_path, doubao_bboxes, qwen_bboxes, doubao_bboxes_18, output_image_path
        )
        
        # 保存bbox信息
        print("正在保存bbox信息...")
        bbox_info_path = save_bbox_info(seq_num, adjusted_doubao_bboxes, adjusted_qwen_bboxes, adjusted_doubao_bboxes_18)
        
        # 清理临时文件，如果使用的是临时下载的图片
        if "temp_image" in temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        print(f"图片处理完成！结果图片已保存为：{output_image_path}")
        print(f"bbox信息已保存为：{bbox_info_path}")
        print(f"豆包模型检测到 {len(adjusted_doubao_bboxes)} 个目标")
        print(f"Qwen模型检测到 {len(adjusted_qwen_bboxes)} 个目标")
        print(f"豆包1.8模型检测到 {len(adjusted_doubao_bboxes_18)} 个目标")
        print(f"=== 第 {i+1} 张图片处理结束 ===")
    
    print("\n所有图片处理完成！")

if __name__ == "__main__":
    main()