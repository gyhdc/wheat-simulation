This repository has made public the key configurations for model training and detection related to the paper "Counting wheat heads using a simulation model" [--], as well as the unified code for evaluating model detection and counting performance.

The `main.py` in the `Assess` folder is the main code for evaluation and test. 

The `Models` folder contains the configuration files for the model during training and test. 

The `Figs` folder contains the code for drawing the main figures and charts in the paper.

During training, test, inference, images will always first be converted to grayscale images.


## Dataset
Dataset could be downloaded from [figshare](https://figshare.com/articles/thesis/Untitled_Item/24198891).    
We use grayscale images for training and testing(images_grayscale folders in the dataset), grayscale images can also be converted from original rgb images through util/togray.py. 

## Trained models
The pretrained YOLOv7 model trained with our simulation wheat images can be downloaded from [Dropbox](https://www.dropbox.com/scl/fi/xhtn1mz1q643i54cf87y4/yolov7_wheat.pt?rlkey=0ah3pxn9k6y49ik9llxai3m39&dl=0)

## Training
1. Clone this repository to your local machine.
2. Download the original YOLOv7 pretrained model from [WongKinYiu's YOLOv7](https://github.com/WongKinYiu/yolov7) and place it in the `weights` folder.
3. In `Models/YOLOV7/data/MakeMyData.yaml`, specify the paths for your training set (processed in grayscale) and validation set (also processed in grayscale).
4. Optionally, set custom parameters (defaults are available).
5. Run `train.py` in `Models/YOLOV7` to start training the model. Expected results should appear within 25-75 epochs.

## Test
1. In `detect.py`, modify custom parameters as needed (defaults are acceptable).
2. Set the path for the test dataset.
3. Run `detect.py`. Results will be available in the `runs/detect` folder.
4. In the `main.py`  inside `Assess`, specify the path for the labels of the test dataset and the labels of detection results.
5. Run `main.py`. Results will be printed in the console.

## Detection (Inference)
1. In `detect.py`, modify custom parameters as needed (defaults are acceptable).
2. Set the path for the test dataset.
3. Run the script for detection. Results will be available in the `runs/detect` folder.
