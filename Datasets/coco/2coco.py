"""
reference1: https://github.com/sisrfeng/aic2coco/blob/master/2coco.py
reference2: https://blog.csdn.net/qq_15969343/article/details/80848175
"""
import json
import os
import cv2


def custom2coco(root_path, phase='instances_val2017', split=400):
    """
    把自定义的数据集转换为coco的数据格式
    Args:
        root_path: 根路径，包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),
                   以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
        phase: 用于创建训练集或验证集，train可以改为val   
        split: 训练集和验证集数目划分的界线
    """
    # dataset是存储最终的结果
    dataset = {'categories':[],'images':[],'annotations':[]}

    # 打开标注文件，在dataset中建立类别标签和数字id的对应关系
    class_path=os.path.join(root_path, 'list/classes.txt')
    with open(class_path) as f:
        classes = f.read().strip().split()
    for i, cls in enumerate(classes, start=1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # 读取train2017文件夹的图片名称，并做了排序处理
    file_path=os.path.join(root_path, 'phase1/train/images')
    file_list=sorted(os.listdir(file_path))
    _names = [f for f in file_list]

    # 判断是建立训练集还是验证集
    if phase == 'instances_train2017':
        names = [line for i, line in enumerate(_names) if i <= split]
    elif phase == 'instances_val2017':
        names = [line for i, line in enumerate(_names) if i > split]

    # 读取Bbox信息，并转换为COCO格式
    with open(os.path.join(root_path, 'list/annos.txt')) as fr:
        annos = fr.readlines()
    for k, name in enumerate(names, 1):
        # 用opencv读取图片，得到图像的宽和高
        print(k, os.path.join(file_path,name))
        im = cv2.imread(os.path.join(file_path,name))
        height, width, _ = im.shape
        
        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': name,
                                'id': k,
                                'width': width,
                                'height': height})
                    
        # 一张图多个框时需要判断
        for ii, anno in enumerate(annos,1):
            parts = anno.strip().split()

            # 如果图像的名称和标记的名称对上，则添加标记
            #             
            if parts[0] == name:
                # 类别
                cls_id = parts[1]
                # x_min
                x1 = float(parts[2])
                # y_min
                y1 = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                x2 = x1 + width
                y2 = y1 + height
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(cls_id),
                    'id': ii,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

    # 保存结果
    # # folder = os.path.join(root_path, 'annotations')
    folder = '/home/projects/competition_orange/data/list/annotations'

    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(folder, '{}.json'.format(phase))
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
    print('done')


if __name__ == "__main__":

    root_path = '/home/projects/competition_orange/data'
    custom2coco(root_path)


