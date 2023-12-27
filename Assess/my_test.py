import math
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

def pianzhi(detectLabelPath=None, labelPath=None):

    detectLabels = os.listdir(detectLabelPath)
    detectLabels = sorted(detectLabels)
    # labels=sorted(os.listdir(labelPath))
    labels = sorted(os.listdir(labelPath))
    def judge_is(x,*l):
        for i in l:
            if x.find(str(i))!=-1:
                return True
        return False

    avg_acc = 0
    test_01 = 'image realLabel detectLabel b\n'
    cnt=0
    pp=0
    for i in range(len(labels)):
        if labels[i] not in detectLabels:
            continue
        pp+=1
        try:
            x=labels[i]
            if judge_is(x,'classes') :
                cnt+=1
                continue
            acc = 0
            testLabel = os.path.join(labelPath, labels[i])
            detectLabel = os.path.join(detectLabelPath, x)

            testNum = len(open(testLabel).readlines())
            detectNum = len(open(detectLabel).readlines())
            acc = 1-(abs(detectNum -testNum)/testNum)
            avg_acc += acc
            test_01 += f'{labels[i]} {testNum} {detectNum} {acc}\n'

        except Exception as e:
            print(i,labels[i],f'\n{e}')
            cnt+=1
            pass
    avg_acc = avg_acc / (len(labels)-cnt)
    test_01 = f"avg_acc\n{avg_acc}\n" + test_01
    # print(pp)
    print('acc:',avg_acc,",")
    # print()
    return test_01

def main_acc(detectLabelPath=None, labelPath=None,path=r"#",name=None):


    currentDT = datetime.datetime.now()
    if name is None:
        file = f"test_{currentDT.hour}-{currentDT.minute}"
        txtPath = os.path.join(path, file + '.txt')
        cnt = 0
        while os.path.exists(txtPath):
            file = '_'.join([file, str(cnt)])
            cnt += 1
            txtPath = os.path.join(path, file + '.txt')
    else:
        file = f"test_{name}"
        txtPath = os.path.join(path, file + '.txt')
        cnt = 0
        while os.path.exists(txtPath):
            file = '_'.join([file, str(cnt)])
            cnt += 1
            txtPath = os.path.join(path, file + '.txt')
    with open(txtPath, mode='w') as f:
        # print(txtPath)
        f.write(pianzhi(detectLabelPath,labelPath))
    data = pd.read_csv(txtPath,header=2,sep=' ')
    x = data.detectLabel
    y = data.realLabel
    rmse = np.sqrt(mean_squared_error(x, y)).round(3)
    mae = mean_absolute_error(x, y).round(3)
    print(f'mae:{mae},rmse:{rmse},')
    return txtPath
