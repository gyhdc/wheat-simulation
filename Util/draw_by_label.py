import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def draw_box(imgPath=None, labelPath=None):
    label_path = labelPath
    image_path = imgPath

    def xywh2xyxy(x, w1, h1, img):
        labels = ['person', 'car']
        label, x, y, w, h = x

        x_t = x * w1
        y_t = y * h1
        w_t = w * w1
        h_t = h * h1



        top_left_x = int(x_t - w_t / 2)
        top_left_y = int(y_t - h_t / 2)
        bottom_right_x = int(x_t + w_t / 2)
        bottom_right_y = int(y_t + h_t / 2)

        image=img
        color = (200,120,120)
        class_name='wheat'
        cv2.rectangle(image, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y),color,  2)
        label = class_name
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (top_left_x, top_left_y - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        new_img=cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
        return new_img


    if not os.path.exists(label_path):
        lb=np.array([])
    else:
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels



    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    for x in lb:

        img=xywh2xyxy(x, w, h, img)
    return img
def crop_wheat(labelsPath,imgPath,saveDir='./temp3'):
    def xywh2xyxy(x,img):

        h1,w1=img.shape[: 2]

        label, x, y, w, h = x

        x_t = x * w1
        y_t = y * h1
        w_t = w * w1
        h_t = h * h1

        top_left_x = int(x_t - w_t / 2)
        top_left_y = int(y_t - h_t / 2)
        bottom_right_x = int(x_t + w_t / 2)
        bottom_right_y = int(y_t + h_t / 2)
        return img[top_left_y:bottom_right_y,top_left_x:bottom_right_x]
    N=300
    labelsDir=random.sample(os.listdir(labelsPath),N)
    total_cnt=0
    for label in tqdm(labelsDir):
        labelName=label
        label=os.path.join(labelsPath,label)
        img = os.path.join(imgPath, labelName.replace('.txt', '.jpg'))
        if not os.path.exists(img):
            img = os.path.join(imgPath, labelName.replace('.txt', '.JPG'))
        image = cv2.imread(img)
        with open(label) as f:
            label_s=f.read().split('\n')
            label_s=random.sample(label_s,min(len(label_s),10))
            for i,l in enumerate(label_s):
                if len(l)<2:
                    continue
                croped_img=xywh2xyxy([float(x) for x in l.split()],image)
                total_cnt+=1
                cv2.imwrite(os.path.join(saveDir,f"{os.path.split(img)[1].split('.')[0]}_{i}.jpg"),croped_img,)
    print(f'保存麦穗 {total_cnt}')




    pass


def clear_folder(folder_path):

    if os.path.exists(folder_path):

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)


            if os.path.isfile(item_path):
                os.remove(item_path)

            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
if __name__ == '__main__':

    if True:

        imgDir=rf"#"
        labelDir=rf"#"
        saveDir=r'./detected'
        savePath=saveDir

        if not os.path.exists(savePath):
            os.mkdir(savePath)
        clear_folder(savePath)

        for img in tqdm(os.listdir(imgDir)):
            fileName=img.split('.')[0]
            imgPath=os.path.join(imgDir, img)
            labelPath=os.path.join(labelDir,fileName+'.txt')
            new_img=draw_box(imgPath,labelPath)
            cv2.imwrite(os.path.join(savePath,img),new_img)