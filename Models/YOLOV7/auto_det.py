import os

from tqdm import tqdm
import subprocess
def getTasks(dirs):
    res=[]
    for dir in os.listdir(dirs):
        res.append(os.path.join(dirs,dir))
    return res

modelPath=r'D:\Desktop\wheat_moni\yolov7-main_1\yolov7-main_5-7_test\yolov7-main\weight'
models=[
    # r"D:\Desktop\wheat_moni\yolov7-main_1\yolov7-main_5-7_test\yolov7-main\weight\1,best.pt",
    # r'D:\Desktop\wheat_moni\yolov7-main_1\yolov7-main_5-7_test\yolov7-main\weight\4.best.pt'
    # r'D:\Desktop\wheat_moni\yolov7-main_1\yolov7-main_5-7_test\yolov7-main\weight\5.best (2).pt'
    r'D:\Desktop\wheat_moni\yolov7-main_1\yolov7-main_5-7_test\yolov7-main\weight\9.best (3).pt'
]
# models=getTasks(r"E:\Datasets\v7-真实数据不同数据量对比\models")
task=[#需要检测的测试集目录
r"E:\Datasets\增强后测试集\gray_blur",
r"E:\Datasets\增强后测试集\gray_downsampled",
r"E:\Datasets\增强后测试集\gray_enhance",
r"E:\Datasets\增强后测试集\gray_reduct",
r"E:\Datasets\增强后测试集\images"
]

# task=getTasks(r'D:\Desktop\wheat_moni\yolov7-main_1\my_util\23test分割日期\res\temp')
task=[
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23-时期分割\green+yellow\green_gray - 副本"
    r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23-时期分割\green+yellow\s\green_gray"
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23-时期分割\5-18\images",
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23-时期分割\5-23\images",
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23-时期分割\6-1\images",
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23-时期分割\6-3\images"
    # r"E:\Datasets\增强后测试集\images",
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23-时期分割\green+yellow\yellow_gray"
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\gwhd\images",
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\21\images",
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\22\images",
    # r"E:\Datasets\论文-模拟小麦数据集\测试集inference\阶段测试集\23\images"

]

import subprocess

# 设置 Conda 环境路径和名称
conda_path = r'E:\Anaconda3'
environment_name = 'yolov7-main_1'

# 构建激活环境的命令
activate_env_command = f'call {conda_path}\\Scripts\\activate {environment_name}'

def getNLevelDirName(path,n=1):
    n-=1
    for i in range(n):
        path=os.path.dirname(path)
    return os.path.split(path)[1].split('.')[0]

# 构建退出环境的命令
deactivate_env_command = f'call {conda_path}\\Scripts\\deactivate'
def detect(mod,imgsDir):
    # dirName=os.path.split(os.path.dirname(imgsDir))[1]
    dirName = getNLevelDirName(imgsDir,n=2)
    # modName=os.path.split(os.path.dirname(mod))[1]
    modName = getNLevelDirName(mod,n=1)
    # 构建运行 Python 脚本的命令
    python_script_command = f'E:\Anaconda3\envs\yolov7-main_1\python.exe detect.py --weights "{mod}" --source "{imgsDir}" --name "{modName}-{dirName}"'
    print(python_script_command)
    os.system(f"{python_script_command}")


def main():
    for mod in models:
        mod=os.path.join(modelPath,mod)
        print(f"检测模型：{mod}")
        for imgs in tqdm(task,desc=f"tasks"):
            detect(mod,imgs)
if __name__ == '__main__':
    main()