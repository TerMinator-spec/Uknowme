# UknoWMe?
In this project an attendance capturing system has been made which takes the image of a person as input and tells whether it is matched with the image of the same person from image database.

## Environment setup
 - Install OpenCV and pytorch  
  ```pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f```  
  ```pip install opencv-python```  
 

## Usage
- Download the img_dtct file as it contains the weights of the trained model.
- Run the process.py file.
- Take two images as input, store their paths and feed the paths to the match function in script.py.
