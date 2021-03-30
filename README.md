# Clothes classifier
## About
- clothes_classifier.py script prints clothing type along with detection confidence based on the input image URL.
- Supports the following clothing types: tshirt, pants, blouse, dress, sweater, coat, hat, boots.
- Supports images with multiple clothes.

## Used model
- https://pjreddie.com/darknet/yolo/
- Trained using https://colab.research.google.com
- colab notebook file: train_clothes_classifier.ipynb
- ~5000 training iterations using 541 images in total

## Prerequisites
- Python 3.6
- Install required Python modules: pip3 install -r requirements.txt
- Download weights from: https://drive.google.com/file/d/1kBcQ_Rh_6qpG5J_RyrASTlpB7Gu4zzZa/view?usp=sharing to the root folder of this repository.

## Usage
- python3 clothes_classifier.py --url='https://image.ceneostatic.pl/data/products/53931474/i-green-top-hat.jpg'
- expected output: hat: 100.00%
- Tested on Ubuntu 20.10.
