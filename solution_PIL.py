# 国教自动化233吴家宝231425070323 WeChat:w18539512122
# 使用Python的PIL库

import json
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 标签数据读取
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# 图像数据下载
def download_images(data, img_folder):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    for img in data['images']:
        response = requests.get(img['coco_url'])
        image = Image.open(BytesIO(response.content))
        image.save(os.path.join(img_folder, f"{img['id']}.jpg"))

# 数据存储格式变换
def transform_data_structure(data):
    image_dict = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_dict:
            image_dict[img_id] = {'image_ids': [], 'category_ids': [], 'bboxes': [], 'segmentations': []}
        image_dict[img_id]['image_ids'].append(img_id)
        image_dict[img_id]['category_ids'].append(ann['category_id'])
        image_dict[img_id]['bboxes'].append(ann['bbox'])
        image_dict[img_id]['segmentations'].append(ann['segmentation'])
    return image_dict

# 可视化图像ID为1000的图像及标签
def visualize_image(image_dict, img_folder, image_id):
    image_path = os.path.join(img_folder, f"{image_id}.jpg")
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    annotations = image_dict[image_id]
    for bbox in annotations['bboxes']:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# 抽取部分数据保存成新的json文件
def save_selected_data(image_dict, selected_ids, output_file):
    selected_data = {'images': [], 'annotations': []}
    for img_id in selected_ids:
        selected_data['images'].append({'id': img_id})
        ann = image_dict[img_id]
        for i in range(len(ann['image_ids'])):
            selected_data['annotations'].append({
                'image_id': ann['image_ids'][i],
                'category_id': ann['category_ids'][i],
                'bbox': ann['bboxes'][i],
                'segmentation': ann['segmentations'][i]
            })
    with open(output_file, 'w') as file:
        json.dump(selected_data, file)

# 示例: 
# data = load_json_data('TestData_coco.json')
# download_images(data, './images')
# image_dict = transform_data_structure(data)
# visualize_image(image_dict, './images', 1000)
# save_selected_data(image_dict, [139, 724, 785, 885, 1000], 'selected_images.json')

