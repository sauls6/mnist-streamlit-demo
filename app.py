import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from streamlit_drawable_canvas import st_canvas # Import the component

# --- Cache model and class names loading ---
@st.cache_resource # Use st.cache_resource for ML models and other global resources
def load_keras_model(model_path):
    try:
        print("Attempting to load Keras model...") # For debugging
        loaded_model = tf.keras.models.load_model(model_path)
        print("Keras model loaded successfully.") # For debugging
        return loaded_model
    except Exception as e:
        # This error will be shown if the initial load fails
        st.error(f"CRITICAL ERROR loading Keras model from '{model_path}': {e}")
        st.error("Ensure the model file is correct, TensorFlow is installed, and the path is valid.")
        st.stop() # Stop the app if the model can't be loaded at all

@st.cache_data # Use st.cache_data for data that, once loaded, doesn't change
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Error: {file_path} not found. Please ensure it's in the same directory.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: {file_path} is not a valid JSON file.")
        st.stop()

# --- Load Model and Class Names using cached functions ---
# Ensure 'mnist_model.keras' is in the same directory as app.py
model = load_keras_model("mnist_model.keras")
class_names = load_json_file('class_names.json')


st.title("MNIST Digit Classifier with Freehand Drawing")
st.write("Draw a digit (0-9) in the box below and click 'Predict Digit'.")

# --- Initialize Session State ---
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = "canvas_initial" # Unique key for the canvas
if 'image_to_predict' not in st.session_state:
    st.session_state.image_to_predict = None

# --- Layout ---
col1, col2 = st.columns([2, 1]) # Canvas column wider

with col1:
    st.subheader("Drawing Canvas")
    # Parameters for the canvas
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 15)
    # Fixed colors
    stroke_color = "#FFFFFF" # White for drawing on black
    bg_color = "#000000"     # Black background
    drawing_mode = "freedraw"
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Not used in freedraw mode for stroke
        stroke_width=stroke_width,
        stroke_color=stroke_color,       # Use fixed white
        background_color=bg_color,       # Use fixed black
        update_streamlit=realtime_update,
        height=280, 
        width=280,
        drawing_mode=drawing_mode,
        key=st.session_state.canvas_key, 
    )

    if canvas_result.image_data is not None:
        img_rgba = canvas_result.image_data.astype('uint8')
        pil_img = Image.fromarray(img_rgba, 'RGBA').convert('L')
        st.session_state.image_to_predict = pil_img

    if st.button("Clear Canvas"):
        st.session_state.canvas_key = f"canvas_{np.random.randint(100000)}"
        st.session_state.image_to_predict = None 
        st.rerun() # Rerun the script to reflect the cleared canvas

with col2:
    st.subheader("Preview & Prediction")
    if st.session_state.image_to_predict is not None:
        st.write("Current Drawing (Grayscale for model):")
        st.image(st.session_state.image_to_predict, width=140)
    else:
        st.info("Draw on the canvas to see a preview.")

    if st.button("Predict Digit"):
        if st.session_state.image_to_predict is not None:
            img_resized = st.session_state.image_to_predict.resize((28, 28))
            img_array = np.array(img_resized).astype('float32')
            img_array /= 255.0
            img_array_reshaped = img_array.reshape(1, 28, 28, 1)

            try:
                # Ensure model is not None before predicting
                if model is not None:
                    prediction = model.predict(img_array_reshaped)
                    predicted_digit_index = np.argmax(prediction)
                    predicted_digit = class_names.get(str(predicted_digit_index), f"Unknown_idx_{predicted_digit_index}")
                    confidence = np.max(prediction) * 100
                    
                    st.success(f"Predicted Digit: **{predicted_digit}**")
                    st.write(f"Confidence: {confidence:.2f}%")
                    
                    if confidence > 90:
                        st.balloons()
                else:
                    st.error("Model is not loaded. Cannot predict.") # Should be caught by cache_resource error handling
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please draw a digit on the canvas first!")

st.sidebar.markdown("---")
st.sidebar.info("Adjust drawing parameters here. The canvas is on the left.")