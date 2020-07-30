# Jetson Nano SSD Retrainer

## Description
This project allows retraining on desktop platform with Nvidia GPU.<br>
In this example Ubuntu 18.04 LTS is used.<br>
Windows is not supported.<br>

## Installing Dependencies

### CUDA Driver

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
```

To download images, do
```
python3 open_images_downloader.py --class-names="CLASSNAMES" --max-images=N
```

### Retraining Models
```
python3 train_ssd --model-dir=models/<MODEL_NAME> --batch-size=N --num-epochs=N
```

### Converting To ONNX Format
```
python3 export_onnx.py --model-dir=models/<MODEL_NAME>
```
