from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import threading

# Function to train a new model
def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(datagen.flow(x_train, y_train, batch_size=64),
              epochs=10,
              validation_data=(x_test, y_test))

    return model

# Start training in a background thread
def start_training():
    if 'model' not in st.session_state:
        st.session_state.model = train_model()

# Streamlit App
st.set_page_config(
    page_title="Tensorflow Model",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.write('# MNIST Digit Recognition')
st.write('## Using a CNN `TensorFlow` model')

st.write('### Draw a digit in 0-9 in the box below')
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)

realtime_update = st.sidebar.checkbox("Update in realtime", True)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
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
    try:
        if 'model' not in st.session_state:
            start_training()

        model = st.session_state.model

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

        mean, std = 0.1307, 0.3081
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

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
