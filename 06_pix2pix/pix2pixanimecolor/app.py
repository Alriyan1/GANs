import os
import time
import cv2
import numpy as np
from PIL import Image

import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Anime Sketch Colorizer",
    page_icon="🎨",
    layout="wide",
)

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:1rem;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.sub{
    color:#BBBBBB;
    font-size:18px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    background:#ff4b4b;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stDownloadButton>button{
    width:100%;
    height:50px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

MODEL_PATH = "pix2pix_generator.h5"

st.sidebar.title("⚙ Settings")

st.sidebar.success("Pix2Pix GAN")

st.sidebar.markdown("---")

st.sidebar.write("### Model")

if os.path.exists(MODEL_PATH):
    st.sidebar.success("Generator Loaded")
else:
    st.sidebar.error("Generator Not Found")

st.sidebar.markdown("---")

st.sidebar.write("### Image Size")
IMG_SIZE = st.sidebar.selectbox(
    "Resolution",
    [256],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.write("### About")

st.sidebar.info(
"""
Pix2Pix GAN converts Anime Sketches
into Colored Anime Images.
"""
)

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


if os.path.exists(MODEL_PATH):
    model = load_model()
else:
    model = None


def predict(image):
    img = image.resize((IMG_SIZE,IMG_SIZE))
    arr = np.array(img).astype(np.float32)

    if arr.shape[-1] == 4:
        arr = arr[:,:,:3]

    arr = arr/127.5 - 1
    arr = np.expand_dims(arr,0)

    pred = model.predict(arr,verbose=0)[0]
    pred = (pred + 1)*127.5

    pred = np.clip(pred,0,255).astype(np.uint8)
    return pred

st.markdown(
'<p class="title">🎨 Anime Sketch Colorizer</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="sub">Pix2Pix GAN Based Anime Image Colorization</p>',
unsafe_allow_html=True
)

st.markdown("---")

uploaded = st.file_uploader(
    "Upload Sketch",
    type=["png","jpg","jpeg"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Input Sketch")

        st.image(image,width='stretch')

    with col2:

        st.subheader("Prediction")

        if st.button("🎨 Colorize"):

            if model is None:

                st.error("Model not found!")

            else:

                with st.spinner("Generating..."):

                    start = time.time()

                    output = predict(image)

                    end = time.time()

                st.image(output,width='stretch')

                st.success(f"Inference Time : {end-start:.3f} sec")

                output_img = Image.fromarray(output)

                save_path = "outputs/result.png"

                output_img.save(save_path)

                with open(save_path,"rb") as f:

                    st.download_button(
                        "⬇ Download Image",
                        f,
                        file_name="anime_colorized.png",
                        mime="image/png"
                    )

st.markdown("---")

st.caption(
"Built with ❤️ using TensorFlow + Pix2Pix + Streamlit"
)