from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

st.set_page_config(
    page_title = "Tensorflow Model",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Function to download the model from Google Drive
def download_model():
    url = 'https://drive.google.com/file/d/1bGKlH7y40zg-IbEr4mhAKYpeHV9a7_p9/view?usp=drive_link'
    output = 'model.keras'
    
    if not os.path.exists(output):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP errors
            with open(output, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    if chunk:  # Filter out keep-alive new chunks
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download model file. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Load the model once
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model('model.keras')

model = load_model()

st.write('# MNIST Digit Recognition')
st.write('## Using a CNN `TensorFlow` model')

st.write('### Draw a digit in 0-9 in the box below')
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)

realtime_update = st.sidebar.checkbox("Update in realtime", True)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    update_streamlit=realtime_update,
    height=300,
    width=300,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert the canvas image to grayscale directly
    input_numpy_array = np.array(canvas_result.image_data)
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image_gs = input_image.convert('L')

    # Convert to numpy array and normalize
    input_image_gs_np = np.asarray(input_image_gs, dtype=np.float32)
    image_pil = Image.fromarray(input_image_gs_np)
    new_image = image_pil.resize((28, 28))
    input_image_gs_np = np.array(new_image) # Resize to 28x28 directly

    tensor_image = np.expand_dims(input_image_gs_np, axis=-1)  # Add channel dimension
    tensor_image = np.expand_dims(tensor_image, axis=0)  # Add batch dimension

    mean, std = 0.1307, 0.3081  # Check if these are correct for your model
    tensor_image = (tensor_image - mean) / std  # Normalize input

    # Perform prediction
    predictions = model.predict(tensor_image)
    output = np.argmax(predictions)
    certainty = np.max(predictions)

    st.write('### Prediction')
    st.write('### ' + str(output))

    st.write('### Certainty')
    st.write(f'{certainty * 100:.2f}%')
    
    st.divider()
    st.write("### Image As a Grayscale `NumPy` Array")
    st.write(input_image_gs_np)

    st.divider()
    st.write("<h7><p>Credits to Ege Güvener / @egegvner</p></h7>", unsafe_allow_html=True)
