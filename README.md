Low Light Image Enhancement

Brightens underexposed photos without needing paired before-and-after training data. Implements Zero-DCE from scratch in TensorFlow/Keras and compares it against a classical CLAHE baseline, wrapped in a Streamlit app you can try in the browser.

Live demo: https://lowlight-zerodce-rithish.streamlit.app

Why this approach

Most image enhancement models learn from pairs: a dark photo and the same photo correctly exposed. Those pairs are expensive to collect and rarely match real shooting conditions.

Zero-DCE sidesteps this. Instead of predicting pixels directly, it predicts a set of curve adjustment maps that get applied to the image iteratively. It learns those curves from four non-reference losses that score the output on its own merits, so no ground truth is ever shown during training.

Loss	What it penalises
Spatial consistency	Neighbouring regions drifting apart in contrast
Exposure control	Regions sitting too far from a target brightness
Color constancy	One channel dominating and casting a tint
Illumination smoothness	Curve maps changing abruptly between pixels
Methods compared

CLAHE (classical). Contrast Limited Adaptive Histogram Equalization, applied per tile so local contrast improves without blowing out already-bright regions. Fast and training-free, but it amplifies noise in very dark frames and cannot recover colour.

Zero-DCE (from scratch). A lightweight CNN predicting per-pixel curve parameters, trained with a custom tf.GradientTape loop rather than model.fit() so the four losses could be weighted and inspected individually. Trained for 100 epochs on 485 images from the LOL dataset.

Running it locally
bash
git clone https://github.com/rithishss/Low-Light-Image-Enhancement.git
cd Low-Light-Image-Enhancement
pip install -r requirements.txt
streamlit run app.py

Then open http://localhost:8501 and upload a dark photo.

Python 3.12 or lower is required. TensorFlow 2.17 publishes no wheel for 3.13, so the install fails outright on newer interpreters.

Deployment notes

Getting this onto Streamlit Cloud's free tier (1 GB container) took some work, and the reasoning is here because the fix is not obvious.

The symptom. Uploading a 12 MP photo killed the container with no traceback. Silent death with no Python error is the signature of the OS out-of-memory killer, not an application bug.

What profiling showed. Peak resident memory scales linearly at roughly 2 KB per output pixel, on top of a fixed ~410 MB TensorFlow baseline:

Longest edge	Output size	Peak RSS
Full resolution	4000 x 3000	4,277 MB
600 px	600 x 450	955 MB
512 px	512 x 384	~845 MB
448 px	448 x 336	725 MB
384 px	384 x 288	621 MB

Why 448 and not 600. TensorFlow does not return memory to the OS after inference. Five sequential runs plateaued at 719 MB rather than falling back to the 410 MB baseline, so the peak becomes the session's floor, not a transient spike. Headroom has to cover the plateau, which rules out anything above roughly 512.

Fixes applied:

Input capped at 448 px on the long edge, aspect ratio preserved
@st.cache_resource on model loading, which was re-reading weights on every rerun
Explicit del and gc.collect() after each inference
10 MB upload cap enforced in .streamlit/config.toml, not just in code. Streamlit buffers the entire upload before the script runs, so an in-app check alone cannot prevent the OOM it targets.
.convert("RGB") during preprocessing. RGBA PNGs previously reached the model as 4 channels and crashed in the first convolutional layer.

Result: 4,277 MB down to 725 MB peak, leaving roughly 290 MB of headroom on a 1 GB container.

The trade-off is sharpness. Images render at around 1120 px, so a 448 px output upscales about 2.5x. MAX_EDGE is a one-line change if you would rather trade headroom for detail.

Limitations
Output resolution is capped at 448 px on the long edge for the hosted demo
Extremely dark inputs still surface sensor noise once brightened
No denoising stage; enhancement and denoising are treated as separate problems here
CPU inference only
Stack

Python, TensorFlow/Keras, OpenCV, NumPy, Pillow, Streamlit
