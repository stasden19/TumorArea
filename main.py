import os

import cv2
from yolo_segmentation import YOLOSegmentation
import numpy as np


def calculate_polygon_area(polylines):
    area = 0

    for polyline in polylines:
        n = len(polyline)
        for i in range(n):
            x1, y1 = polyline[i]
            x2, y2 = polyline[(i + 1) % n]  # Замыкаем полигон

            area += (x1 * y2 - x2 * y1)

    return abs(area / 2.0)


def calculate_tumor_area(img_path):
    if not os.path.exists(img_path):
        raise FileExistsError('File not found')
    img_new = cv2.imread(img_path)
    img = cv2.resize(img_new, (640, 640), fx=0.7, fy=0.7)
    ys = YOLOSegmentation("best.pt")
    bboxes, classes, segmentations, scores = ys.detect(img)
    all_res = list(zip(bboxes, classes, segmentations, scores))
    for bbox in bboxes:
        (x, y, x2, y2) = bbox
        cv2.line(img, ((x + x2) // 2, y), ((x + x2) // 2, y2), (255, 123, 87), 2)
        dlina = x2 - x
        cv2.imshow("image", img)
        cv2.waitKey(0)
        scale = int(input("Измерьте и укажите длину показанной линии в миллиметрах: ")) / dlina
        scale *= scale
        break
    img_new = cv2.imread(img_path)
    img = cv2.resize(img_new, (640, 640), fx=0.7, fy=0.7)
    all_area = 0

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        (x, y, x2, y2) = bbox
        area = int(calculate_polygon_area([seg]))
        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.polylines(img, [seg], True, (0, 0, 255), 4)
        cv2.putText(img, str(round(area * scale, 2)), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        all_area += round(area * scale, 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    return all_area


height1 = float(input("Введите рост пациента при первичном осмотре: "))
weight1 = float(input("Введите вес пациента при первичном осмотре: "))
tumor_image1 = input("Загрузите изображение опухоли при первичном осмотре: ")
tumor_area1 = calculate_tumor_area(tumor_image1)
print("Площадь опухоли при первичном осмотре:", tumor_area1, "мм²")

# Получаем данные при повторном осмотре
height2 = float(input("Введите рост пациента при повторном осмотре: "))
weight2 = float(input("Введите вес пациента при повторном осмотре: "))
tumor_image2 = input("Загрузите изображение опухоли при повторном осмотре: ")
tumor_area2 = calculate_tumor_area(tumor_image2)
print("Площадь опухоли при повторном осмотре:", tumor_area2, "мм²")

# Вычисляем S1, S2, S тела 1 и S тела 2
S1 = tumor_area1 / height1
S2 = tumor_area2 / height2
S_body_1 = height1 * weight1
S_body_2 = height2 * weight2
absolute_res = (S2 - S1 / S1 - S_body_2 - S_body_1 / S_body_1) * 100
if absolute_res > 10:
    print(True)
else:
    print(False)
