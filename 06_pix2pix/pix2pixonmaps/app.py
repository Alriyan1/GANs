import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2
import time

st.set_page_config(
    page_title='Pix2Pix Map to Satellite',
    page_icon='🛰️',
    layout='wide'
)

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E3A8A;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

div.stButton > button{
    width:100%;
    height:50px;
    font-size:18px;
    border-radius:12px;
}

img{
    border-radius:12px;
}

.metric{
    background:#ffffff;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)


st.markdown('<p class="title"> Pix2Pix Map -> Satellite Generator</p>',unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate Satellite Images using Pix2Pix GAN</p>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return ort.InferenceSession('generator_model3.onnx')


session = load_model()

def preprocess(image):
    image = image.resize((256,256))
    image = np.array(image).astype(np.float32)
    image = image/127.5-1.0
    image = np.expand_dims(image,0)
    return image

def postprocess(output):
    output = output.squeeze()
    output = (output +1)/2
    output = np.clip(output,0,1)
    output = (output*255).astype(np.uint8)
    return Image.fromarray(output)


st.sidebar.title("⚙ Model Information")

st.sidebar.info("""
**Model**

Pix2Pix GAN

**Framework**

TensorFlow

**Inference**

ONNX Runtime

**Input Size**

256 × 256

**Output**

Satellite Image
""")


uploaded = st.file_uploader(
    'Upload Map Image',
    type=['png','jpg','jpeg']
)

if uploaded:

    image = Image.open(uploaded).convert('RGB')
    col1,col2 = st.columns(2)

    with col1:
        st.subheader('Original Map')
        st.image(image,width='stretch')

    if st.button('Generate Satellite'):
        start = time.time()
        inp = preprocess(image)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        output = session.run(
            [output_name],
            {input_name:inp}
        )[0]

        pred = postprocess(output)

        end = time.time()

        with col2:
            st.subheader('Generated Satellite')
            st.image(pred,width='stretch')

        st.success('Generated Completed')
        c1,c2,c3 = st.columns(3)

        c1.metric("Inference Time",f"{end-start:.3f} sec")

        c2.metric("Input Size","256 × 256")

        c3.metric("Output Size","256 × 256")

        st.download_button(
            "⬇ Download Image",
            data=np.array(pred).tobytes(),
            file_name="generated_satellite.png",
            mime="image/png"
        )

else:

    st.info("Upload a map image to begin.") 