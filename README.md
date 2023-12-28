This repository has made public the key configurations for model training and detection related to the paper "Counting wheat heads using a simulation model" [ ], as well as the unified code for evaluating model detection and counting performance.

The `main.py` in the `Assess` folder is the main code for evaluation testing. 

The `Models` folder contains the configuration files for the model during training and testing. 

The `Figs` folder contains the code for drawing the main figures and charts in the paper.





## Dataset
- Download the dataset from [figshare](https://figshare.com/articles/thesis/Untitled_Item/24198891).

## Training
1. Clone this repository to your local machine.
2. Download the pre-trained YOLOv7 model from [WongKinYiu's YOLOv7](https://github.com/WongKinYiu/yolov7) and place it in the `weights` folder.
3. In `Models/YOLOV7/data/MakeMyData.yaml`, specify the paths for your training set (processed in grayscale) and validation set (also processed in grayscale).
4. Optionally, set custom parameters (defaults are available).
5. Use `train.py` in `Models/YOLOV7` to start training the model. Expected results should appear within 25-75 epochs.

## Detection
1. In `detect.py`, modify custom parameters as needed (defaults are acceptable).
2. Set the path for the test dataset.
3. Run the script for detection. Results will be available in the `runs/detect` folder.

## Evaluation
1. In the `main.py`  inside `Assess`, specify the path for the test dataset labels and the labels of the detected results.
2. Perform the evaluation. Results will be printed in the console.
