import os
import shutil

import cv2
from PIL.ImageEnhance import Color
from tqdm import tqdm
import numpy as np
def toGray(imgPath, labelPath, savePath):
    import cv2 as cv
    import shutil
    from tqdm import tqdm
    def img2Gray(filePath, savePath):
        image = cv.imread(filePath)
        gray2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imwrite(savePath, gray2)

    imgSavePath = os.path.join(savePath)
    labelSavePath = os.path.join(savePath, 'labels')
    if os.path.exists(imgSavePath) == False:
        os.mkdir(imgSavePath)

    imgs = os.listdir(imgPath)

    if labelPath != '#':
        labels = os.listdir(labelPath)
        for label in tqdm(labels, desc='label'):
            oldLabel = os.path.join(labelPath, label)
            newLabel = os.path.join(labelSavePath, label)

            shutil.copyfile(oldLabel, newLabel)
    for img in tqdm(imgs, desc='images'):
        try:
            oldImg = os.path.join(imgPath, img)
            newImg = os.path.join(imgSavePath, img)
            img2Gray(oldImg, newImg)
        except:
            pass
def clean_dir(Path):
    for file in tqdm(os.listdir(Path), desc=f'clear：{Path}'):
        try:
            file = os.path.join(Path, file)
            os.remove(file)
        except:
            pass

if __name__ == '__main__':
    imageDir = r"#"
    saveDir = r"./temp"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    clean_dir(saveDir)
    labelDir = r"./save"
    toGray(
            imgPath=imageDir,
            labelPath='#',
            savePath=saveDir
           )

