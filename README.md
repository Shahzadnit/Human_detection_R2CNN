## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Perform training on Penn-Fudan for Pedestrian Detection dataset (https://www.cis.upenn.edu/~jshi/ped_html/) converted in COCO format for training. Dataset is placed in directory 'Human_detection_R2CNN/datasets/human' 

**1. start training. weight will save in directory 'R2CNN/output/human/' or we can change the path in config file 'Human_detection_R2CNN/configs/Human_det_train.yaml' in line no. 68**
````
python tools/train_net.py --config-file "configs/Human_det_train.yaml"
````


## Inference on Penn-Fudan for Pedestrian Detection dataset. Download trained model
**1. Download [model](https://drive.google.com/file/d/12BT745oHv8lvy7HtBpZJgmbUVMIo_RFe/view?usp=sharing) and put in drectory 'Human_detection_R2CNN/output/human/'** 

**2. Single image inference. result will save 'Human_detection_R2CNN/datasets/human/result'**
````
cd ./tools
python inference_engine.py
````


