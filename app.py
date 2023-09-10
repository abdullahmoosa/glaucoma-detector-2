import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import streamlit_authenticator as stauth

# Load the pre-trained Glaucoma detection model
# from keras.models import model_from_json
# json_file = open('models\model (1).json')
# loaded_model_json = json_file.read()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(r"C:\Users\abdul\developement\Python-projects\streamlit-glaucoma-detection\glaucoma_detector\models\ResNet50_weights.h5")
# print("Loaded model from disk")
st.set_page_config(
        page_title="Glaucoma Detector",
)


import yaml
from yaml.loader import SafeLoader

# Load the YAML file


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
model = tf.keras.models.load_model(r'C:\Users\abdul\developement\Python-projects\streamlit-glaucoma-detection\glaucoma_detector\models\raw-test1.h5', compile = False)
# Define a function to make predictions on the input image
def predict_glaucoma(image):
    # Preprocess the image (resize, normalize, etc. based on your model's requirements)
    processed_image = preprocess_image(image)

    # Make prediction
    # prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]
    pred = model.predict(np.expand_dims(processed_image, axis=0))[0]

    return pred

# Define a function to preprocess the input image
def preprocess_image(image):
    # Implement any necessary preprocessing (resize, normalization, etc.)
    # Example:
    processed_image = cv2.resize(image, (224, 224))  # Resize to match model input size


    return processed_image

# Streamlit app with enhanced styling
def main():
    # Add content for the authenticated page    
    # Authenticate the user
    name, authentication_status, username = authenticator.login('Login', 'main')
    if authentication_status:
        st.title("Glaucoma Detector")
        st.markdown("**Accuracies of Various Models :**")
        data = {'Models': ['CNN', 'Resnet50', 'MobileNetv2','Vgg16','InceptionV3'],
            'Accuracy': [68.702, 50.382, 68.103,58.779,72.591],
            'f1 score' : [70.021, 48.921, 71.087,52.661,72.471],
            'precision score' : [68.891, 52.032, 69.041,67.153,72.591],
            'recall score' : [68.802, 51.021, 70.023,58.775,72.553]}
        df = pd.DataFrame(data)
        df.set_index('Models', inplace=True)
        df.index.name = 'Models'
        st.table(df)
        st.markdown("**Confusion Matrix of best model:**")
        st.image("models\download (4).png", caption='Confusion Matrix', use_column_width=True)
        
        # Add a header
        st.header("Upload an Image")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Read the uploaded image
            # image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            image = uploaded_file.read()
            processed_image = cv2.imdecode(np.fromstring(image, np.uint8), 1)

            # Display the uploaded image with caption
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        
            # Make prediction
            prediction = predict_glaucoma(processed_image)
        
            # Display the prediction
            st.subheader("Prediction")
            # st.write(f"Probability: {prediction[0]:.2f}")
            if prediction >0.5:
                st.error("Glaucoma detected.")
            else:
                st.success("No Glaucoma detected.")

    elif authentication_status == False:
        st.error("Authentication failed. Please try again.")
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__ == '__main__':
    main()
