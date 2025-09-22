import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras.backend as K
from keras.models import load_model
import time
fig = plt.figure()

st.title('Image-Based Classification of Plant Diseases')

st.markdown("Pick an image of a plant to discover if it is healthy or not and in particular which is the disease it suffers of:")


def custom_categorical_crossentropy(y_true, y_pred):
    '''This is the weighted categorical crossentropy I implemented in the notebook. 
    When the model is uploaded the custom loss function has to be uploaded with it in using custom object.'''

    normalized_weights = np.load('weights.npy')        #these are the weights computed in the notebook 
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred) * normalized_weights
    loss = -K.sum(loss, -1)
    return loss


def main():
    '''Function that defines the main components of the app interface.'''

    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])     #interface object that allows to upload an image file.
    class_btn = st.button("Classify")    #button that makes the classification start.
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image") #error message shown when the button "classify" is pressed without uploading an image.
        else:
            with st.spinner('Model working....'): #if an image is actually uploaded, then the predict function (which implements the logical part of the application) is called.
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)



def predict(image):
    '''Function that implements the logic of the application and that actually computes the predictions of the model'''

    classifier_model = "saved_CNN_model.h5" #underlying CNN Keras model trained on the notebook with Horovod with Spark.
    model = load_model(classifier_model, custom_objects={'custom_categorical_crossentropy': custom_categorical_crossentropy})
    test_image = image.resize((64,64)) #the input image is resized, transformed in a numpy array, scaled and then reshaped to respect the network input requirements.
    test_image = keras.utils.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.reshape(test_image, (-1, 12288))
    class_names = [
          'Bread_Bean_healthy',
          'Bread_bean_Botrytis',
          'Cherry___Powdery_mildew',
          'Cherry___healthy',
          'Corn___Common_rust',
          'Corn___healthy',
          'Grape___Esca_(Black_Measles)',
          'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
          'Grape___healthy',
          'Olive_healthy',
          'Olive_peacocks_eye_(Spilocaea_oleaginea)',
          'Onion_White_rot',
          'Onion_healthy',
          'Peach___Bacterial_spot',
          'Peach___healthy',
          'Pepper__bell___Bacterial_spot',
          'Pepper__bell___healthy',
          'Potato___Early_blight',
          'Potato___healthy',
          'Strawberry_Leaf_spot',
          'Strawberry___Leaf_scorch',
          'Strawberry___healthy',
          'Tomato___Bacterial_spot',
          'Tomato___Late_blight',
          'Tomato___Leaf_Mold',
          'Tomato___Spider_mites-Two-spotted_spider_mite',
          'Tomato___Target_Spot',
          'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
          'Tomato___Tomato_mosaic_virus',
          'Tomato___healthy'] 
    predictions = model.predict(test_image) #computes the different classification probabilities for that image.
    scores = tf.nn.softmax(predictions[0]) 
    scores = scores.numpy()

    result = f"{class_names[np.argmax(scores)]}" #returns the label of the class with the highest probability. 
    return result


if __name__ == "__main__":
    main()

