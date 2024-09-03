from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import pandas as pd

st.set_page_config(
    page_title="Tensorflow Model",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
)

time.sleep(0.1)
tf.keras.backend.clear_session()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('EMNIST_byClass_Model.keras')

model = load_model()

data = {
    'Layer': ['1', '2', '3', '4', '5'],
    'Neurons': [128, 256, 256, 256, 10]
}

st.write('# MNIST Digit Recognition')
st.write('###### Using a CNN `TensorFlow` Model')

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

    st.write("# Model Analysis")
    st.write("###### Since Last Update")

    st.write("##### \n")

    col1, col2, col3 = st.columns(3)
    
    col1.metric(label="Epochs", value=10, delta=9, help="One epoch refers to one complete pass through the entire training dataset.")

    col2.metric(label="Accuracy", value="96.76%", delta="0.52%", help="Total accuracy of the model which is calculated based on the test data.")

    col3.metric(label="Model Train Time", value="0.18h", delta="0.4h", help="Time required to fully train the model with specified epoch value. (in hours)", delta_color="inverse")

    st.divider()
    st.write("# Number of Neurons")
    st.write("# ")
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index('Layer'), x_label="Layer Number", y_label="Neurons")
    st.write("# ")

    st.write("Softmax activation function in layer 5:")
    st.latex(r"softmax({z})=\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}")
    
    st.divider()

    st.markdown("""
<img src="https://www.cutercounter.com/hits.php?id=hxpcokn&nd=9&style=1" border="0" alt="website counter"></a>
""", unsafe_allow_html=True)
st.write("###### Credits to `Ege Güvener`/ `@egegvner` @ 2024")
