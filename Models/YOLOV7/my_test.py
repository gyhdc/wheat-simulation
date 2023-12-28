import math
import os
labelPath=r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\21小麦_100\labels"
detectLabelPath=r"D:\Desktop\wheat_moni\yolov7-main_1\yolov7-main_5-7_test\yolov7-main - 副本\runs\detect\exp7\labels"
path= r"D:\Desktop\wheat_moni\yolov7-main_1\TEST\test_14_负面"
detectLabels=os.listdir(detectLabelPath)
detectLabels=sorted(detectLabels)
# labels=sorted(os.listdir(labelPath))
labels=sorted(os.listdir(labelPath))

def avg_acc_():


    avg_acc=0
    test_01='image realLabel detectLabel acc\n'
    p=0
    for i in range(len(labels)):
        if labels[i] not in detectLabels:
            continue
        p+=1
        try:
            acc=0
            testLabel=os.path.join(labelPath,labels[i])
            detectLabel=os.path.join(detectLabelPath,labels[i])

            testNum=len(open(testLabel).readlines())
            detectNum=len(open(detectLabel).readlines())
            acc=detectNum/testNum
            avg_acc+=acc
            test_01+=f'{labels[i]} {testNum} {detectNum} {acc}\n'
        except:

            pass
    avg_acc=avg_acc/len(labels)
    test_01=f"avg_acc\n{avg_acc}\n"+test_01

    return test_01
def pianzhi():
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
    print(pp)
    print(avg_acc)
    print()
    return test_01
def pianzhi_test():
    def judge_is(x,*l):
        for i in l:
            if x.find(str(i))!=-1:
                return True
        return False

    avg_acc = 0
    test_01 = 'image realLabel detectLabel b\n'
    cnt=0
    for i in range(len(labels)):
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
    return test_01
def avg_pz_1(labels,detectedDir):
    detectedLabels=sorted(os.listdir(detectedDir),key=lambda x:int(x[2:-5]))
    print(detectedLabels)
    avg_acc=0
    test_01 = 'image realLabel detectLabel b\n'
    lcnt=[int(x) for x in open(labels).readlines()]
    for i,dlabel in enumerate(detectedLabels):
        o_d=dlabel
        dlabel=os.path.join(detectedDir,dlabel)
        # print(dlabel)
        dcnt=len(open(dlabel).readlines())
        l_cnt=int(lcnt[i])
        # print(dcnt,l_cnt)
        acc = 1 - (abs(dcnt - l_cnt) / l_cnt)
        avg_acc += acc
        test_01 += f'{o_d} {l_cnt} {dcnt} {acc}\n'
    avg_acc = avg_acc / len(lcnt)
    test_01 = f"avg_acc\n{avg_acc}\n" + test_01
    print(avg_acc)


    return test_01


if __name__ == '__main__':
    import datetime

    currentDT = datetime.datetime.now()
    file=f"test_{currentDT.hour}-{currentDT.minute}"
    txtPath=os.path.join(path,file+'.txt')
    cnt=0
    while os.path.exists(txtPath):
        file='_'.join([file,str(cnt)])
        cnt+=1
        txtPath=os.path.join(path, file+'.txt')
    with open(txtPath,mode='w') as f:
        print(txtPath)
        f.write(pianzhi())
        # f.write(avg_pz_1(
        #     r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\22\22.txt",
        #     r"D:\Desktop\wheat_moni\yolov7-main_1\yolov7-main_5-7_test\yolov7-main\runs\detect\22\labels"
        # ))