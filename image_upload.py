import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json
import io

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        original_image = Image.open(uploaded_file)
        original_image = np.array(original_image)
        return original_image
    else:
        return None
    

        
def load_model():
    json_file = open('model (2).json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    
    #zip_file = "model_best_weights.zip"
    #file_to_extract = "weights.h5"
 
    #try:
        #with zipfile.ZipFile(zip_file) as z:
            #with open(file_to_extract, 'wb') as f:
                #f.write(z.read(file_to_extract))
                #print("Extracted", file_to_extract)
    #except:
        #print("Invalid file")
    loaded_model.load_weights("weights.h5")
    return loaded_model

def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

def predict(loaded_model, categories, x):
    labels = categories
    width = 224
    height = 224
    dim = (width, height)
    #import numpy as np 
    #image = np.array(image)
    #img = cv2.resize(image, dim,interpolation = cv2.INTER_LINEAR)

    #img = image.load_img(target_size=(224, 224))
    #x = image.img_to_array(image)
    x = cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    x = np.expand_dims(x, axis=0) /255
    classes = loaded_model.predict(x)
    st.subheader('Predictions')
    st.write("Closed:", np.argmax(classes[0])==0, (classes[0][0]), round(classes[0][0]) * 100,"%") # 0 : Eyes closed
    st.write("Open:" , np.argmax(classes[0])==1, (classes[0][1]), round(classes[0][1]) * 100,"%") # 1 : Eyes open
    st.write("No-yawn:" , np.argmax(classes[0])==2, (classes[0][2]), round(classes[0][2])  * 100,"%")# 2 : No yawn
    st.write("Yawn:" , np.argmax(classes[0])==3, (classes[0][3]), round(classes[0][3])  * 100,"%")# 3 : Yawn
    st.write(f"Predicted label: {labels[np.argmax(classes[0])]}") # Predicted label
    st.write(f"Probability of prediction): {round(np.max(classes[0])) * 100} %") # Probablilty of prediction
    st.subheader('Conclusion')
    if (np.argmax(classes[0])==0 or np.argmax(classes[0])==3):
        st.write("Drowsy")
    else:
        st.write("Not drowsy")




def main():
    
    LABELS_PATH = 'model_classes.txt'
    st.title('Drowsy Detection Demo')
    st.header('A transfer learning approach')
    model = load_model()
    categories = load_labels(LABELS_PATH)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image)


if __name__ == '__main__':
    main()
