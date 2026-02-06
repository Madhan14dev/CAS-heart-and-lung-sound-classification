import streamlit as st
import tempfile
import json
from inference import predict_murmur, THRESHOLD

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Murmur Classification",
    layout="centered"
)

st.title("🫀 Heart Sound Murmur Classification")
st.markdown(
    "Upload a **heart sound (.wav)** file to detect murmur presence "
    "using a Whisper-based deep learning model."
)

# --------------------------------------------------
# CONFIG INFO
# --------------------------------------------------
with open("config.json", "r") as f:
    config = json.load(f)

st.sidebar.header("Model Configuration")
st.sidebar.write(f"**Sample Rate:** {config['sample_rate']} Hz")
st.sidebar.write(f"**Max Duration:** {config['max_duration_sec']} sec")
st.sidebar.write(f"**Decision Threshold:** {config['decision_threshold']}")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Heart Sound (.wav)",
    type=["wav"]
)

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav"
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing heart sound..."):
        result = predict_murmur(tmp_path)

    st.success("Inference Completed")

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Murmur Probability",
            value=result["murmur_probability"]
        )

    with col2:
        st.metric(
            label="Prediction",
            value=result["prediction"]
        )

    st.info(
        f"Classification based on threshold = {THRESHOLD}"
    )
