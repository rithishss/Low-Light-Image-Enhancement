import gc
import os
import streamlit as st
from PIL import Image
from zerodce import ZeroDCE
import tensorflow as tf
import numpy as np
import keras

# Weights live next to this file, so resolve them relative to it rather than
# the working directory Streamlit happens to be launched from.
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zerodce1.h5")

# The free Streamlit Cloud tier caps a container at roughly 1 GB, and
# TensorFlow plus the model already claim ~410 MB of it. Inference memory
# scales with pixel count, so shrink the image before it reaches the model: a
# 12 MP phone photo at full resolution peaks around 4.3 GB and gets the
# container OOM-killed mid-session (no traceback, the process just dies).
#
# Measured peak RSS of the running server on a 12 MP upload, by long edge:
#     384px -> 621 MB    448px -> 728 MB    512px -> ~845 MB    600px -> 955 MB
# TensorFlow does not hand memory back to the OS afterwards, so the peak is
# also the session's floor. 448 keeps ~290 MB of headroom; raising it is a
# one-line change if the app ever moves off the free tier.
MAX_EDGE = 448
MAX_UPLOAD_MB = 10

st.set_page_config(page_title="Low Light Image Detection", page_icon="📷", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
    }
    .larger-heading {
        font-size: 24px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    show_home()

@st.cache_resource(show_spinner="Loading the ZeroDCE model...")
def load_model():
    """Build and load the model once per container, not on every rerun."""
    model = ZeroDCE()
    model.load_weights(WEIGHTS_PATH)
    return model

def prepare_image(uploaded_file):
    """Decode an upload as RGB, shrunk so its long edge is at most MAX_EDGE.

    Returns the prepared image alongside the original dimensions so the UI can
    say when it resized something.
    """
    image = Image.open(uploaded_file)
    original_size = image.size  # header only; nothing is decoded yet
    # draft() lets PIL decode a JPEG straight to a smaller size, so the
    # full-resolution pixels never exist in memory. No-op for other formats.
    image.draft("RGB", (MAX_EDGE, MAX_EDGE))
    # ZeroDCE expects 3 channels; without this an RGBA png would reach the
    # model as 4 and blow up in the first conv layer.
    image = image.convert("RGB")
    image.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    return image, original_size

def infer(original_image, model):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    array = output_image.numpy()
    result = Image.fromarray(array)
    # The float32 tensors dwarf the PIL images; drop them before returning
    # rather than waiting for them to fall out of scope.
    del image, output_image, array
    gc.collect()
    return result

def zeroimage(original_image):
    return infer(original_image, load_model())

def show_home():
    st.markdown("<h1 class='centered-title'>Low Light Image Detection 📷</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='larger-heading'>Upload Image</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload a low-light image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_image:
        size_mb = uploaded_image.size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(
                f"That file is {size_mb:.1f} MB, which is over the {MAX_UPLOAD_MB} MB limit. "
                "Please upload a smaller image."
            )
        else:
            original_image, original_size = prepare_image(uploaded_image)
            st.image(original_image, caption="Uploaded Image", width="stretch")
            if original_size != original_image.size:
                st.caption(
                    f"Resized from {original_size[0]}x{original_size[1]} to "
                    f"{original_image.size[0]}x{original_image.size[1]} to stay within "
                    "the memory available on Streamlit Cloud."
                )

            output_image = zeroimage(original_image)
            st.image(output_image, caption="ZeroDCE", width="stretch")

            del original_image, output_image
            gc.collect()
        # Simulating returned images for demonstration
#        returned_images = [
#            {"title": "ZeroDCE", "path": "output_image"},
#            {"title": "NafNet", "path": "path_to_nafnet_image"},
#            {"title": "CLAHE", "path": "path_to_clahe_image"},
#            {"title": "Sambhav", "path": "path_to_sambhav_image"}
#        ]

    st.caption("Built by Rithish S · [github.com/rithishss](https://github.com/rithishss)")

if __name__ == "__main__":
    main()
