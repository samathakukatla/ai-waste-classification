import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Waste Classification System",
    page_icon="???",
    layout="centered"
)

st.title("?? AI Waste Classifier System")
st.write("Upload an image of waste to classify it into its correct category.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_model.h5")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Model file waste_classifier_model.h5 not found. Please train the model first.")

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Classify Waste"):
        with st.spinner("Analyzing image..."):
            img = image.resize((224, 224))
            img_array = np.array(img)
            
            if img_array.ndim == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
                
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = 100 * np.max(predictions[0])

            st.markdown(f"### Result: **{predicted_class.upper()}**")
            st.progress(float(np.max(predictions[0])))
            st.info(f"Confidence Level: {confidence:.2f}%")
