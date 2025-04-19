import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pickle
from tensorflow.keras.models import model_from_json

# Load CNN model dari JSON dan weights
with open("CNN_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

cnn_model = model_from_json(loaded_model_json)
cnn_model.load_weights("best_model.weights.h5")

# Load scaler dan encoder
with open("scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pickle", "rb") as f:
    encoder = pickle.load(f)
    emotion_labels = list(encoder.categories_[0])  # urutan label dari encoder

# Ekstraksi fitur (ZCR, RMSE, MFCC)
def extract_features_for_cnn(uploaded_file):
    y, sr = sf.read(uploaded_file)
    y = y[:int(sr * 2.5)]  # ambil 2.5 detik

    frame_length = 2048
    hop_length = 512

    def zcr(data):
        return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

    def rmse(data):
        return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

    def mfcc(data, sr):
        mfcc_feat = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=30)
        return np.ravel(mfcc_feat.T)

    features = np.hstack((zcr(y), rmse(y), mfcc(y, sr)))

    if features.shape[0] != 2376:
        features = features[:2376] if features.shape[0] > 2376 else np.pad(features, (0, 2376 - features.shape[0]), mode='constant')

    scaled_features = scaler.transform(features.reshape(1, -1))
    return np.expand_dims(scaled_features, axis=2)  # shape: (1, 2376, 1)

# Streamlit App
st.set_page_config(page_title="Prediksi Emosi Suara", page_icon="🎧", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎧 Emotion Recognition from Speech</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload file audio dan dapatkan prediksi emosinya!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Unggah file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍 Prediksi Emosi"):
        with st.spinner("🔄 Sedang memproses dan memprediksi..."):
            features = extract_features_for_cnn(uploaded_file)
            prediction = cnn_model.predict(features)[0]
            predicted_label = emotion_labels[np.argmax(prediction)]

            st.markdown(f"<h2 style='text-align: center; color:#00bfff;'>💡 Emosi Terdeteksi: <span style='color:#ff4b4b'>{predicted_label.upper()}</span></h2>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📊 Skor Keyakinan Model:")

            for label, score in zip(emotion_labels, prediction):
                st.progress(float(score))
                st.markdown(f"**{label.capitalize()}**: {score:.4f}")

