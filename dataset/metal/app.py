import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="centered"
)

st.title(" AI Waste Classification System")
st.write("Upload an image of waste to classify it into its correct category.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_model.h5")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Model file 'waste_classifier_model.h5' not found. Please train the model first.")

# Category labels matching your folder names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# File uploader widget
uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Waste"):
        with st.spinner("Analyzing image..."):
            # Preprocess image
            img = image.resize((224, 224))
            img_array = np.array(img)
            
            # Ensure image has 3 color channels (RGB)
            if img_array.ndim == 2:  # Grayscale
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]
                
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = 100 * np.max(predictions[0])

            # Show results
            st.markdown(f"### Result: **{predicted_class.upper()}**")
            st.progress(float(np.max(predictions[0])))
            st.info(f"Confidence Level: {confidence:.2f}%")