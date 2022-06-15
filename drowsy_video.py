# import libraries
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

#Load the saved model
model = tf.keras.models.load_model('weights.h5')

# runs on .mp4 video
video = cv2.VideoCapture('nhtsa.mp4')

# runs on webcam
#video = cv2.VideoCapture(0)


while True:
        _, frame = video.read()
        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        #Resizing into 224x224 
        im = im.resize((224,224))
        img_array = im#image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        
        prediction = np.argmax(probabilities)
        
        # if the model detects eyes closed or yawning, convert RBG to B&W
        if prediction == 0 or prediction == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                #print(prediction)
                print(probabilities[prediction])
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        # terminate program when 'q' is pressed
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()