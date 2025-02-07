import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Helper function to load and predict plant disease
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = Image.open(test_image).resize((128, 128))  # Use PIL to read and resize the image
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar navigation
st.sidebar.title("Smart Farming with AI: Predicting Potato Leaf Diseases for Better Yields")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition', 'About the Model', 'Agricultural Tips', 'Contact Us'])

# Home Page
if app_mode == 'Home':
    img = Image.open('desktop-wallpaper-the-future-of-agriculture-organic-farming.jpg')
    st.image(img)
    st.markdown("<h1 style='text-align:center;'>Plant Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Potato Leaf Disease Plant Detection System designed to help farmers identify potato plant diseases quickly and efficiently.")

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Potato Leaf Disease Plant Detection System For Sustainable Agriculture')

    test_image = st.file_uploader('Choose an image:', type=['jpg', 'jpeg', 'png'])
    
    # Show the uploaded image
    if test_image and st.button('Show Image'):
        st.image(test_image, use_column_width=True)

    # Make a prediction
    if test_image and st.button('Predict'):
        st.snow()
        st.write('Our Prediction:')
        try:
            result_index = model_prediction(test_image)
            class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
            st.success(f'Model is predicting it\'s a {class_name[result_index]}')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# About the Model Page
elif app_mode == 'About the Model':
    st.header('About the Plant Disease Detection Model')
    st.write("This model is trained using TensorFlow and is capable of detecting various potato plant diseases based on image inputs. The model has been fine-tuned for optimal accuracy and is designed to assist farmers in early disease detection.")
    st.write("The classes recognized by the model are:")
    st.markdown("""
    - **Potato Early Blight**
    - **Potato Late Blight**
    - **Healthy Potato Plant**
    """)
    st.info("Please ensure high-quality images for accurate predictions.")

# Agricultural Tips Page
elif app_mode == 'Agricultural Tips':
    st.header('Agricultural Tips for Better Yield')
    st.write("Here are some tips to maintain healthy potato crops:")
    st.markdown("1. **Soil Preparation:** Ensure well-drained, nutrient-rich soil.")
    st.markdown("2. **Disease Monitoring:** Regularly inspect plants for early signs of disease.")
    st.markdown("3. **Proper Irrigation:** Maintain adequate soil moisture but avoid overwatering.")
    st.markdown("4. **Crop Rotation:** Rotate crops to reduce disease buildup.")
    st.markdown("5. **Pest Control:** Use organic methods to control pests effectively.")
    st.info("Implementing these tips can significantly improve your crop yield.")

# Contact Us Page
elif app_mode == 'Contact Us':
    st.header('Contact Us')
    st.write("We are here to assist you with any queries regarding plant disease detection and agricultural tips.")
    st.markdown("- **Email:** pranaydhore03@gmail.com")
    st.markdown("- **Phone:** +91-7498678112")
