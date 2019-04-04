# import the necessary packages
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import time
import sys
import os
import cv2
from PIL import Image
import numpy as np
import keras
from keras import layers
from keras import models

frame_index=0
video = cv2.VideoCapture(0)

# load model
from keras.models import load_model 
model = load_model('traffic_sign_e100.h5')
model.summary()

# use signnames.csv to obtain mappings from the class id (integer) to the actual sign name
import csv

signnames=[] # containing the sign names, read from the csv file
with open('signnames.csv', 'rt') as csvfile:
    signreader = csv.reader(csvfile)
    for row in signreader:
        signnames.append(row[1])
    signnames.remove(signnames[0]) # remove the header


# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

for i in range(5):
	img_path = "/data2/traffic-signs-data-gtsrb/new_test_photo/"+str(i+1).zfill(2)+".png"
	img = image.load_img(img_path, target_size=(32, 32))
	img_tensor = image.img_to_array(img) # shape will be (32,32,3)
	img_tensor = np.expand_dims(img_tensor, axis=0) # shape will be (1,32,32,3)
	# Remember that the model was trained on inputs
	# that were preprocessed in the following way:
	img_tensor /= 255. # scale to [0..1] in float32 type

	# Its shape is (1, 150, 150, 3)
	output_pred = model.predict(img_tensor) # Prob. as confidence of predicted classes
	predicted_label = np.argmax(output_pred)
	print(np.max(output_pred)) # Print out the most confidence output
	print(predicted_label)		
	print(signnames[predicted_label]) # Print out the name of label
	print()

"""
while True:
	frame_index+=1
	_, frame = video.read()

	width = frame.shape[1]
	# crop frame: center: 480x480 and resize to 32x32
	frame = frame[:, 320-480//2:320+480//2]

	dim = (32,32)
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	#print(frame.shape, frame.dtype)
	#print(frame)
	frame_tensor = np.expand_dims(frame, axis=0) # shape will be (1,480,480,3)
	#frame_tensor.astype(np.float32)
	frame_tensor = frame_tensor / 255.0 # normalize input as [0..1]
	#print(frame_tensor.dtype, frame_tensor.shape)

	output_pred = model.predict(frame_tensor) # Prob. as confidence of predicted classes
	predicted_label = np.argmax(output_pred)
	print(np.max(output_pred)) # Print out the most confidence output
	print(predicted_label)		
	print(signnames[predicted_label]) # Print out the name of label

	# Append text on image
	output = "Stop_Sign"
	cv2.putText(frame, output, (480//2, 480//2), cv2.FONT_HERSHEY_COMPLEX, 1 , (0,255,255), 2)

    # show the frame
	cv2.imshow("Frame", frame)

	# save image
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	im=Image.fromarray(frame)
	#print(type(im))

	im.save(os.path.join("data2", str(frame_index).zfill(6)+".jpg"), "JPEG")

	key = cv2.waitKey(1) & 0xFF 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

video.release()
cv2.destroyAllWindows()
"""