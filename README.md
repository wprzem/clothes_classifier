# Clothes classifier
## About
- clothes_classifier.py script prints clothing type along with detection confidence based on the input image URL.
- Supports the following clothing types: tshirt, pants, blouse, dress, sweater, coat, hat, boots.
- Supports images with multiple clothes.


- 541 images used to train the model.
- Trained for ~5000 iterations.


## Prerequisites
- Download weights from: https://drive.google.com/file/d/1kBcQ_Rh_6qpG5J_RyrASTlpB7Gu4zzZa/view?usp=sharing to the root folder of this repository.

## Usage
- python3 clothes_classifier.py --url='https://image.ceneostatic.pl/data/products/53931474/i-green-top-hat.jpg'
- expected output: hat: 100.00%
