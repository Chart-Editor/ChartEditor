from cnocr import CnOcr
import numpy as np
import cv2
from ultralytics import YOLO
import os
print(os.cpu_count())

# 路径 输入
yolo_path_save = '/hpc2hdd/home/syan195/pipeline/test-pie'
image_path = '/hpc2hdd/home/syan195/pipeline/input_files/pie_sales_distribution_sports_2.png'
yolo_model = "/hpc2hdd/home/syan195/ultralytics/runs/segment/train7/weights/best.pt"
# user_input = "I want a bar chart that fits the style used to describe a rocket launch."

image = cv2.imread(image_path)
height, width, channels = image.shape
img_shape = (height, width)

# yolo分割
model = YOLO(yolo_model,task='segment') 
model.predict(source=image_path,save=True,show=True,save_txt=True,project=yolo_path_save)

yolo_txt = os.path.join(yolo_path_save+'/predict18'+'/labels/'+os.path.splitext(os.path.basename(image_path))[0]+'.txt')

# ocr进行文字部分的补充分割
ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')
out = ocr.ocr(image_path)

def save_ocr_results_to_txt(ocr_results, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in ocr_results:
            position = item['position'].tolist()  # 将 numpy array 转换为列表
            score = item['score']
            text = item['text']
            # 写入文件，格式为：position, score, text
            f.write(f"Position: {position}\n")
            f.write(f"Score: {score}\n")
            f.write(f"Text: {text}\n")
            f.write("\n")  # 每个块之间用空行分隔
    print('finish write')

save_ocr_results_to_txt(out, '/hpc2hdd/home/syan195/pipeline/test-pie/ocr_results.txt')

def convert_to_percentage(coords, img_shape):
    """将位置坐标转换为百分比"""
    h, w = img_shape[:2]
    
    # 确保 coords 是一个 NumPy 数组并且是二维的
    coords = np.array(coords)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    
    return [coord[0] / w for coord in coords], [coord[1] / h for coord in coords]

def is_position_within_box(position, box_coords):

    """检查 position 是否在给定的 box 范围内"""
    x_min, y_min = np.min(position, axis=0)
    x_max, y_max = np.max(position, axis=0)
    
    box_x_min, box_y_min = np.min(box_coords, axis=0)
    box_x_max, box_y_max = np.max(box_coords, axis=0)

    return x_min >= box_x_min and x_max <= box_x_max and y_min >= box_y_min and y_max <= box_y_max

def calculate_iou(box1, box2):
    """
    计算两个框的交并比（IoU）。
    
    :param box1: 第一个框的坐标，格式为 [x_min, y_min, x_max, y_max]
    :param box2: 第二个框的坐标，格式为 [x_min, y_min, x_max, y_max]
    :return: 交并比（IoU）
    """
    # 计算相交区域的坐标
    print(box1)
    print(box2)

    x_min_inter = np.maximum(box1[0], box2[0])
    y_min_inter = np.maximum(box1[1], box2[1])
    x_max_inter = np.minimum(box1[2], box2[2])
    y_max_inter = np.minimum(box1[3], box2[3])

    # 计算相交区域的面积
    inter_area = np.maximum(0, x_max_inter - x_min_inter) * np.maximum(0, y_max_inter - y_min_inter)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算交并比（IoU）
    iou = inter_area / (box1_area + box2_area - inter_area)
    print(iou)

       # 确保 iou 是一个标量
    if np.isscalar(iou):
        return iou
    else:
        return iou.item()  # 将单元素数组转换为标量值

def convert_to_box(position):
    """
    将 position 转换为边界框坐标，格式为 [x_min, y_min, x_max, y_max]
    """
    x_percent,y_percent = position
    x_min = min(x_percent) 
    y_min = min(y_percent) 
    x_max = max(x_percent) 
    y_max = max(y_percent) 
    return [x_min, y_min, x_max, y_max]

def process_detections(out, yolo_txt, img_shape):
    with open(yolo_txt, 'r') as file:
        yolo_lines = file.readlines()

    new_annotations = []

    for detection in out:
        position = detection['position']
        text = detection['text']
        is_class_4 = False
        # 转换为百分比
        x_percent, y_percent = convert_to_percentage(position, img_shape)
         # 将 position 转换为 box 坐标
        detection_box = convert_to_box((x_percent, y_percent))

        # 检查是否在 class=5, 6, 7 的范围内，如果是，跳过处理
        skip_detection = False
        for line in yolo_lines:
            class_id, *coords = map(float, line.split())
            if int(class_id) in [5, 6, 7]:
                box_coords = np.array(coords).reshape(-1, 2)
                box_x_min = np.min(box_coords[:, 0])
                box_y_min = np.min(box_coords[:, 1])
                box_x_max = np.max(box_coords[:, 0])
                box_y_max = np.max(box_coords[:, 1])
                yolo_box = [box_x_min, box_y_min, box_x_max, box_y_max]

                # 计算 IoU
                iou = calculate_iou(detection_box, yolo_box)

                # 判断 IoU 是否大于 0.5
                if iou > 0.5:
                    skip_detection = True
                    break

        if skip_detection:
            continue  # 跳过该 detection

        # 检查是否在class为4的范围内
        for line in yolo_lines:
            class_id, *coords = map(float, line.split())
            if int(class_id) == 4:
                box_coords = np.array(coords).reshape(-1, 2)
                box_x_min, box_y_min = np.min(box_coords, axis=0)
                box_x_max, box_y_max = np.max(box_coords, axis=0)

                 # 判断 position 是否在 box 的范围内
                if is_position_within_box(np.array([x_percent, y_percent]).T, box_coords):
                    new_class = 11
                    is_class_4 = True
                    break
            
        if not is_class_4:
            # 如果不在class 4范围内，根据chart类型处理
            for line in yolo_lines:
                class_id, *coords = map(float, line.split())
                if int(class_id) == 1:  # bar chart
                    bar_coords = np.array(coords).reshape(-1, 2)
                    bar_x_min, bar_y_min = np.min(bar_coords, axis=0)
                    bar_x_max, bar_y_max = np.max(bar_coords, axis=0)
                    
                    if np.max(x_percent) > bar_x_max:  # position 在bar chart下面
                        new_class = 9  # x-tick
                    elif np.max(y_percent) < bar_y_min:  # position 在bar chart左侧
                        new_class = 8  # y-tick/
                    else:
                        continue
                elif int(class_id) == 3:  # pie chart
                    new_class = 12  # pie chart label
                else:
                    continue
        # 写入新的标注
        new_annotations.append(f"{new_class} " + " ".join(f"{x} {y}" for x, y in zip(x_percent, y_percent)))

    # 将新的标注写入txt文件
    with open(yolo_txt, 'a') as file:
        file.write("\n".join(new_annotations) + "\n")

process_detections(out, yolo_txt, img_shape)
