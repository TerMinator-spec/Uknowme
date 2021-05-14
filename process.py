import torch
# Load the model
model=my_model(3)
model.load_state_dict(torch.load("/content/gdrive/MyDrive/img_dtct"))

threshold_loss=0.0030/1012

# function to visualize the image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
w=256
h=256
def show_img(path):
    img = Image.open(path) # use pillow to open a file
    img = img.resize((w, h)) # resize the file to 256x256
    img = img.convert('RGB') #convert image to RGB channe
    img = np.asarray(img)/255
    plt.imshow(img)

# Function to get the image converted to torch tensor    
def get_img(path):
  img = Image.open(path) # use pillow to open a file
  img = img.resize((w, h)) # resize the file to 256x256
  img = img.convert('RGB') #convert image to RGB channe
  img = np.asarray(img)/255
  img = np.asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
  img = torch.from_numpy(np.asarray(img))
  return img    

# Evaluate the results matched or unmatched
def eval(path1, path2):
  xx = get_img(path1)
  yy = get_img(path2)
  img_x = xx.reshape(1, 3,w, h)
  img_y = yy.reshape(1, 3,w, h)
  x1 = model(img_x.float())
  y1 = model(img_y.float())
  # calculate loss
  criterion = MSELoss()
  loss = criterion(x1,y1)
  if(loss<=threshold_loss):
    print("Matched \n")
  else:
    print('Unmatched \n')
  print(loss)
  print("\n")
  
# Function which takes two input images and tell weather they matched or not
def match(im1,im2):

  img1 = Image.open(im1) # use pillow to open a file
  img1 = img1.resize((w, h)) # resize the file to 256x256
  img1 = img1.convert('RGB') #convert image to RGB channe
  img1 = np.asarray(img1)/255
  #same for image 2
  img2 = Image.open(im2) # use pillow to open a file
  img2 = img2.resize((w, h)) # resize the file to 256x256
  img2 = img2.convert('RGB') #convert image to RGB channe
  img2 = np.asarray(img2)/255

  f, axarr = plt.subplots(1,2)
  axarr[0].imshow(img1)
  axarr[1].imshow(img2)
  plt.show()
  eval(im1,im2)  
