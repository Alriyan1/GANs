import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import time

st.set_page_config(
    page_title="Pix2Pix Edge ➜ Shoe Generator",
    page_icon="👟",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#0e1117;
}

.block-container{
    padding-top:2rem;
}

h1,h2,h3{
    color:white;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:18px;
    border-radius:12px;
}

.metric{
    background:#262730;
    padding:10px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        provider = ['CUDAExecutionProvider']
    else:
        provider = ['CPUExecutionProvider']

    session = ort.InferenceSession(
        'generator_model.onnx',providers=provider
    )

    return session


session = load_model()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(image):
    image = image.convert('RGB')
    image = image.resize((256,256))
    image = np.array(image).astype(np.float32)
    image = image/127.5-1.0
    image = image.transpose(2,0,1)
    image = np.expand_dims(image,0)
    return image


def postprocess(output):

    img = output[0]

    print("Output Shape:", img.shape)

    # (1,C,H,W)
    if len(img.shape) == 4:
        img = img[0]
        img = img.transpose(1, 2, 0)

    # (C,H,W)
    elif len(img.shape) == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    # (H,W,C)
    elif len(img.shape) == 3 and img.shape[-1] == 3:
        pass

    else:
        raise ValueError(f"Unexpected output shape: {img.shape}")

    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


st.sidebar.title("⚙ Model Information")

st.sidebar.success("Pix2Pix Generator")

st.sidebar.write("Framework : ONNX Runtime")

st.sidebar.write(f"Input Tensor : {input_name}")

st.sidebar.write(f"Output Tensor : {output_name}")

st.sidebar.write("Input Size : 256 × 256")

st.sidebar.write(
    "Provider : "
    + session.get_providers()[0]
)

st.title("👟 Edge ➜ Shoe Generator")

st.write(
    "Upload an edge image and generate a realistic shoe using the Pix2Pix ONNX model."
)

uploaded = st.file_uploader('Upload Edge Image',type=['png','jpg','jpeg'])

if uploaded:

    image = Image.open(uploaded)
    col1,col2 = st.columns(2)

    with col1:
        st.subheader('Input Edge')
        st.image(image,width='stretch')

    if st.button('Generate Shoe'):

        inp = preprocess(image)
        start = time.time()

        output = session.run(
            [output_name],
            {input_name:inp}
        )

        end = time.time()

        result = postprocess(output)

        with col2:
            st.subheader('Generated Shoe')
            st.image(result,width='stretch')

        st.success(f"Inference Time: {(end-start)*1000:.2f} ms")

        st.download_button(
            "⬇ Download Image",
            cv2.imencode('.png',cv2.cvtColor(result,cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name='generated_shoe.png',
            mime='image/png'
        )