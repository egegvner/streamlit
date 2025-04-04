from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import pandas as pd

st.set_page_config(
    page_title="Ege AI",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
)

time.sleep(0.1)
tf.keras.backend.clear_session()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

model = load_model()

data = {
    'Layer': ['1', '2', '3', '4', '5'],
    'Neurons': [256, 512, 512, 256, 10]
}

def load_comments():
    try:
        return pd.read_csv('comments.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['Name', 'Comment'])

def save_comment(name, comment):
    comments = load_comments()
    new_comment = pd.DataFrame({'Name': [name], 'Comment': [comment]})
    comments = pd.concat([comments, new_comment], ignore_index=True)
    comments.to_csv('comments.csv', index=False)

st.write("# MNIST Digit Recognition Using a Tensorflow `Keras` Model")

st.write('#### Draw a digit `(0 - 9)` below')

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

    st.write(f'#### Certainty: \v`{certainty * 100:.2f}%`')
    st.write("###### Model Version `4.3.5`")
    st.write("###### Parameters `1,880,458`")
    st.divider()
    st.write("### Image As a Grayscale `NumPy` Array")
    st.write(input_image_gs_np)

    st.write("# Model Analysis")
    st.write("###### Since Last Update")

    st.write("##### \n")

    col1, col2, col3 = st.columns(3)
    
    col1.metric(label="Epochs", value=50, delta=40, help="One epoch refers to one complete pass through the entire training dataset.")

    col2.metric(label="Accuracy", value="99.34%", delta="2.39%", help="Total accuracy of the model which is calculated based on the test data.")

    col3.metric(label="Model Train Time", value="0.33h", delta="0.12h", help="Time required to fully train the model with specified epoch value. (in hours)", delta_color="inverse")

    st.divider()
    st.write("# Number of Neurons")
    st.write("# ")
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index('Layer'), x_label="Layer Number", y_label="Neurons")
    
    st.divider()

    st.write("###### Credits to / Model by `Ege Güvener` / `@egegvner` Sept, @2024")

    st.write("# Leave a Comment")

    name = st.text_input('Name')
    comment = st.text_area('Comment')

    if st.button('Submit', type = "primary"):
        if name and comment:
            save_comment(name, comment)
            st.success('Comment submitted successfully!')
        elif name == "":
            st.error("Don't you have a name?")
        elif comment == "":
            st.error("Why would you post an empty comment?")

    st.subheader('Existing Comments')
    comments = load_comments()
    for i, row in comments.iterrows():
        st.write(f"**{row['Name']}**: {row['Comment']}")
