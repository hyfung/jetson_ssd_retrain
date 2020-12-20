# Jetson Nano SSD Retrainer

## Description
This project allows retraining on desktop platform with Nvidia GPU.<br>
In this example Ubuntu 18.04 LTS is used.<br>
Windows is not supported.<br>

Retrained models are stored in https://github.com/hyfung/jetson_ssd_retrain_models

## Installing Dependencies

### CUDA Driver
Ubuntu 18.04 LTS is used in this demo
```
cd ~/Downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

### Python3 Packages
You should install necessary Python packages by doing
```
pip3 install -r requirements.txt
```
The packages used are: pytorch, pandas, boto3

## Usage

### Downloading Datasets From Open Images
Refer to https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt for labels

To view dataset stats, do
```
python3 open_images_downloader.py --stats-only --class-names="CLASSNAMES"
#For example
#python3 open_images_downloader.py --stats-only --class-names="Man,Woman"
```

To download images, do
```
python3 open_images_downloader.py --class-names="CLASSNAMES" --max-images=N
#For example
#python3 open_images_downloader.py --class-names="Man,Woman" --max-images=2500
```

List of useful dataset
* Man,Woman
* Bus,Car,Motorcycle,Truck,Van,Taxi
* Human head,Human body,Human arm,Human hand,Human leg,Human foot
* Human face,Human eye,Human ear,Human nose,Human mouth
* Stop sign,Traffic light,Traffic sign,Vehicle
* Vehicle,Vehicle registration plate


### Retraining Models
Training a basic model (mb1-ssd)

Other options are: mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd
```
python3 train_ssd.py --model-dir=models/<MODEL_NAME> --batch-size=N --num-epochs=N
#For example
#python3 train_ssd.py --model-dir=models/man_woman --batch-size=8 --num-epochs=100

#Lazy guy's shortcut
python3 train_ssd.py --model-dir=models/ --batch-size=8 --num-epochs=50
```

To train VGG16-SSD
```
python3 train_ssd.py --net=vgg16-ssd --pretrained-ssd=models/vgg16-ssd-mp-0_7726.pth --model-dir=models/<MODEL_NAME> --batch-size=N --num-epochs=N
```

### Converting To ONNX Format
```
python3 onnx_export.py --model-dir=models/<MODEL_NAME>
#For example
#python3 onnx_export.py --model-dir=models/man_woman

#Lazy guy's shortcut
python3 onnx_export.py --model-dir=models/
```

### Testing Models with SSD (Static Image)
```
python3 run_ssd_example.py mb1-ssd *.pth models/*/labels.txt test.jpg
```

### Testing Models with SSD (Live)
```
python3 run_ssd_example_live.py mb1-ssd *.pth models/*/labels.txt 0
```

Human Hand

![alt text](https://github.com/hyfung/jetson_ssd_retrain/blob/master/images/01.png "")

Dog

![alt text](https://github.com/hyfung/jetson_ssd_retrain/blob/master/images/02.png "")

Vehicles

![alt text](https://github.com/hyfung/jetson_ssd_retrain/blob/master/images/03.png "")

## Other Pretrained Models
Visit https://github.com/hyfung/jetson_ssd_retrain_models for my pretrained other models
