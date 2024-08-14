from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title = "Tensorflow Model By Ege",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
)

model = tf.keras.models.load_model('model1.keras')

st.write('# MNIST Digit Recognition')
st.write('## Using a CNN `TensorFlow` model')

st.write('### Draw a digit in 0-9 in the box below')

epochs = st.sidebar.slider("Epochs", 1, 10, 1)

if st.sidebar.button("Load Model", type="primary"):
    model = tf.keras.models.load_model(f"model{epochs}.keras")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    input_numpy_array = np.array(canvas_result.image_data)
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image_gs = input_image.convert('L')

    input_image_gs_np = np.asarray(input_image_gs, dtype=np.float32)
    image_pil = Image.fromarray(input_image_gs_np)
    new_image = image_pil.resize((28, 28))
    input_image_gs_np = np.array(new_image)

    tensor_image = np.expand_dims(input_image_gs_np, axis=-1)
    tensor_image = np.expand_dims(tensor_image, axis=0)

    mean, std = 0.1307, 0.3081
    tensor_image = (tensor_image - mean) / std

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
else:
    pass
