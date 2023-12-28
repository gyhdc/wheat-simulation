import os

from tqdm import tqdm
import subprocess
def getTasks(dirs):
    res=[]
    for dir in os.listdir(dirs):
        res.append(os.path.join(dirs,dir))
    return res

modelPath=r'#'
models=[
    r'#'
]


task=[
    r"testset path"
]

import subprocess


conda_path = r'#'


def getNLevelDirName(path,n=1):
    n-=1
    for i in range(n):
        path=os.path.dirname(path)
    return os.path.split(path)[1].split('.')[0]


def detect(mod,imgsDir):

    dirName = getNLevelDirName(imgsDir,n=2)

    modName = getNLevelDirName(mod,n=1)

    python_script_command = f'{conda_path} --weights "{mod}" --source "{imgsDir}" --name "{modName}-{dirName}"'
    print(python_script_command)
    os.system(f"{python_script_command}")


def main():
    for mod in models:
        mod=os.path.join(modelPath,mod)
        print(f"running：{mod}")
        for imgs in tqdm(task,desc=f"tasks"):
            detect(mod,imgs)
if __name__ == '__main__':
    main()