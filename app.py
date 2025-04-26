import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd

# Load the trained model
model = joblib.load("model/genre_classifier.pkl")

# Genre labels
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Emojis for genres
genre_emojis = {
    'blues': '🎼', 'classical': '🎻', 'country': '🤠', 'disco': '🪩',
    'hiphop': '🎤', 'jazz': '🎷', 'metal': '🤘', 'pop': '🎧',
    'reggae': '🟢', 'rock': '🎸'
}

# Page config
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

st.title("🎶 Music Genre Classifier")
st.markdown("Drag & drop a `.wav` file and I’ll guess the genre with confidence!")

# File uploader (Drag & Drop)
file = st.file_uploader("🎵 Drop your `.wav` file here", type=["wav"], label_visibility="collapsed")

if file:
    with st.spinner("🔍 Analyzing audio... please wait"):
        # Audio preview
        st.audio(file, format='audio/wav')

        # Load audio
        y, sr = librosa.load(file, duration=30)

        # Feature extraction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Tempo estimate
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        st.markdown(f"**🎚 Estimated Tempo (BPM):** {tempo_val:.2f}")

        # Tempo-based hint
        if tempo_val < 80:
            st.info("🧘 Chill tempo — could be classical, blues, or jazz.")
        elif tempo_val > 120:
            st.info("💥 Fast tempo — could be metal, dance, or hip hop.")

        # Genre prediction
        prediction = model.predict(mfcc_mean)[0]
        probs = model.predict_proba(mfcc_mean)[0]
        confidence = np.max(probs) * 100

        # Result
        emoji = genre_emojis.get(prediction, '')
        st.markdown(f"### 🎯 Predicted Genre: **{prediction.upper()}** {emoji}")
        st.markdown(f"**📈 Confidence Score:** {confidence:.2f}%")

        # Top 3 genre probabilities
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_data = {
            "Genre": [genres[i].capitalize() for i in top3_idx],
            "Probability (%)": [probs[i]*100 for i in top3_idx]
        }
        df_top3 = pd.DataFrame(top3_data)

        st.markdown("### 📊 Top 3 Genre Probabilities")
        st.bar_chart(df_top3.set_index("Genre"))

        # Optional waveform
        with st.expander("📈 Show waveform preview"):
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr)
            ax.set_title('Waveform')
            st.pyplot(fig)


