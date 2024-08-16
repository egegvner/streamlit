from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time

try:

    st.set_page_config(
        page_title = "Tensorflow Model by Ege G.",
        page_icon="💎",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    time.sleep(0.01)
    tf.keras.backend.clear_session()
    time.sleep(0.01)

    model = tf.keras.models.load_model('model.keras')

    st.write('# MNIST Digit Recognition')
    st.write('###### Using a CNN `TensorFlow / Keras` Model')

    st.write('#### Draw a digit in 0-9 in the box below')

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

        st.write(f'# Prediction: \v`{str(output)}`')

        st.write(f'##### Certainty: \v`{certainty * 100:.2f}%`')
        
        st.divider()
        st.write("### Image As a Grayscale `NumPy` Array")
        st.write(input_image_gs_np)

        st.divider()

        st.write("# Model Analysis")
        st.write("###### Since Last Update")

        st.write("##### \n")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(label="Epochs", value=10, delta=9, help="One epoch refers to one complete pass through the entire training dataset.")

        col2.metric(label="Accuracy", value="98.53%", delta="0.26%", help="Total accuracy of the model which is calculated based on the test data.")

        col3.metric(label="Model Train Time", value="0.16h", delta="0.08h", help="Time required to fully train the model with specified epoch value. (in hours)", delta_color="inverse")

    st.divider()
    st.write("###### Handmade model by / credits to `Ege Güvener`/ `@egegvner` @ 2024")
except Exception as e:
    st.write("# *Refresh the page if you encounter any errors.")
    st.write("##### Keep reloading until non-error output is produced.\n")
    st.error(f"Error: {e}")
