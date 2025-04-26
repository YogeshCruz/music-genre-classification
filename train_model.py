import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Replace with your GTZAN dataset path
DATASET_PATH = "genres_original"

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

X, y = [], []

for genre in genres:
    folder_path = os.path.join(DATASET_PATH, genre)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            try:
                y_audio, sr = librosa.load(file_path, duration=30)
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                features = np.mean(mfcc.T, axis=0)
                X.append(features)
                y.append(genre)
            except:
                continue

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, "model/genre_classifier.pkl")
print("✅ Model saved to model/genre_classifier.pkl")
