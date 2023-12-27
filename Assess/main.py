
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

def intersection_area(label_box, detect_box):
    x_label_min, y_label_min, x_label_max, y_label_max = label_box
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_box
    if (x_label_max <= x_detect_min or x_detect_max < x_label_min) or (
            y_label_max <= y_detect_min or y_detect_max <= y_label_min):
        return 0
    else:
        lens = min(x_label_max, x_detect_max) - max(x_label_min, x_detect_min)
        wide = min(y_label_max, y_detect_max) - max(y_label_min, y_detect_min)
        return lens * wide


def union_area(label_box, detect_box):
    x_label_min, y_label_min, x_label_max, y_label_max = label_box
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_box

    area_label = (x_label_max - x_label_min) * (y_label_max - y_label_min)
    area_detect = (x_detect_max - x_detect_min) * (y_detect_max - y_detect_min)
    inter_area = intersection_area(label_box, detect_box)

    area_union = area_label + area_detect - inter_area

    return area_union


def getIOU(label_bbox, detect_bbox):
    i_area = intersection_area(label_bbox, detect_bbox)
    u_area = union_area(label_bbox, detect_bbox)
    iou = i_area / u_area
    return iou


def xywh2xyxy(xywh, w1, h1):
    xywh = [float(item) for item in xywh]
    x, y, w, h = xywh
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    top_left_x = int(x_t - w_t / 2)
    top_left_y = int(y_t - h_t / 2)
    bottom_right_x = int(x_t + w_t / 2)
    bottom_right_y = int(y_t + h_t / 2)
    return [top_left_x, top_left_y, bottom_right_x, bottom_right_y]


def getScores(imgsPath, labelPath, detectPath, iou_threshold):
    # print(imgsPath,labelPath)
    with open(labelPath) as fl:

        labels = [x.strip().split()[1:] for x in fl.readlines() if len(x) > 2]
    with open(detectPath) as fd:
        detecteds = [x.strip().split()[1:] for x in fd.readlines() if len(x) > 2]
    img = cv2.imread(imgsPath)
    h, w = img.shape[:-1]
    labels = [
        xywh2xyxy(xywh, w1=w, h1=h) for xywh in labels
    ]
    detecteds = [
        xywh2xyxy(xywh, w1=w, h1=h) for xywh in detecteds
    ]
    tp_lst = []
    fp_lst = []
    for detected in detecteds:
        ok = False
        for label in labels:
            iou = getIOU(label, detected)
            if ok == True:
                break
            if iou > iou_threshold:
                tp_lst.append(detected)
                ok = True
        if ok == False:
            fp_lst.append(detected)
    return {
        'tp': len(tp_lst),
        "fp": len(fp_lst),
        "fn": abs(len(labels) - len(tp_lst))
    }


def calPrecisionAndRecall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def accScores(imgsDir, labelsDir, detectedDir, iou_threshold):
    if not isinstance(imgsDir, str):
        imgsDir = imgsDir[0]
    if not os.path.isdir(imgsDir) and 1==0:
        res = getScores(
            imgsDir,
            labelsDir,
            detectedDir,
            iou_threshold=iou_threshold
        )
        tp, fp, fn = res['tp'], res["fp"], res['fn']

    else:
        labels = os.listdir(labelsDir)
        detecteds = os.listdir(detectedDir)
        imgs = os.listdir(imgsDir)
        tp, fp, fn = 0, 0, 0
        for img in tqdm(imgs):
            image = os.path.join(imgsDir, img)
            label = os.path.join(labelsDir, img.split('.')[0] + '.txt')
            detected = os.path.join(detectedDir, img.split('.')[0] + '.txt')
            res = getScores(image, label, detectPath=detected, iou_threshold=iou_threshold)
            tp += res['tp']
            fp += res['fp']
            fn += res['fn']
    print(f"tp:{tp},fp:{fp},fn:{fn}")
    classify_acc=tp/(tp+fp+fn)
    print("classify_acc=",classify_acc)
    return calPrecisionAndRecall(tp, fp, fn)


def calF1(p, r):
    f1 = 2 * p * r / (p + r + 1e-16)
    return f1


if __name__ == '__main__':

    is_train = False

    imgsDirs=[
        r"#"#test-img1

    ]
    detectedDir=r"#"
    def getDetLabels(dir,target,labelDirName="labels"):
        return [
            os.path.join(detectedDir,file,labelDirName) for file in os.listdir(dir) if file.find(target) !=-1
        ]

    detectedDirs=[

        [
            # os.path.join(detectedDir,"labels") for file in os.listdir(detectedDir) if file.find('gwhd') !=-1
        ],
        [
            # os.path.join(detectedDir,"labels") for file in os.listdir(detectedDir) if file.find('21') !=-1
        ],
        [
            # os.path.join(detectedDir,"labels") for file in os.listdir(detectedDir) if file.find('22') !=-1
        ],
        [
            # os.path.join(detectedDir,"labels") for file in os.listdir(detectedDir) if file.find('23') !=-1
        ],

    ]
    for idx,imgsDir in enumerate(imgsDirs):

        for detectedDir in detectedDirs[idx]:
            labelsDir=os.path.join(os.path.split(imgsDir)[0],'labels')
            pres=[]
            recs=[]
            f1s=[]
            ious=[0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
            ious=[0.5]
            name = f"{os.path.split(os.path.dirname(detectedDir))[1]}"
            # print(name)
            for iou in ious:
                pre, rec = accScores(
                    imgsDir=imgsDir,
                    labelsDir=labelsDir,
                    detectedDir=detectedDir,
                    iou_threshold=iou
                )
                print(name)
                print(f"iou:{iou}  ,pre:{pre}, rec:{rec},f1:{calF1(pre, rec)},")

                pres.append(pre)
                recs.append(rec)
                f1s.append(calF1(pre, rec))

            from my_test import  main_acc
            if not os.path.exists('./test'):
                os.mkdir("./test")
            accPath=main_acc(detectLabelPath=detectedDir,labelPath=labelsDir,
                             path=r"./test",
                             name=f"{os.path.split(os.path.dirname(detectedDir))[1]}_ano"
                             )
            from test import main
            main(accPath)
            print("-")

