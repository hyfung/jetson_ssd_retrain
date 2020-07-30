# Jetson Nano SSD Retrainer

## Description
This project allows retraining on desktop platform with Nvidia GPU.<br>
In this example Ubuntu 18.04 LTS is used.<br>
Windows is not supported.<br>

## Installing Dependencies

### CUDA Driver
Ubuntu 18.04 LTS is used in this demo
```
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

### Retraining Models
```
python3 train_ssd --model-dir=models/<MODEL_NAME> --batch-size=N --num-epochs=N
#For example
#python3 train_ssd --model-dir=models/man_woman --batch-size=8 --num-epochs=100
```

### Converting To ONNX Format
```
python3 export_onnx.py --model-dir=models/<MODEL_NAME>
#For example
#python3 export_onnx.py --model-dir=models/man_woman
```
