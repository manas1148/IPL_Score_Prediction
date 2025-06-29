import streamlit as st
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load data
ipl = pd.read_csv('ipl_data.csv')
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis=1)

# Prepare encoders and scaler
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

X = df.drop(['total'], axis=1)
y = df['total']

X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define and train the model (or load pre-trained model if available)
model = keras.Sequential([
    keras.layers.Input(shape=(X_scaled.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
huber_loss = tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam', loss=huber_loss)
model.fit(X_scaled, y, epochs=10, batch_size=64, verbose=0)  # Fewer epochs for demo

# Streamlit UI
st.title('IPL Score Predictor')

venue = st.selectbox('Select Venue', df['venue'].unique())
batting_team = st.selectbox('Select Batting Team', df['bat_team'].unique())
bowling_team = st.selectbox('Select Bowling Team', df['bowl_team'].unique())
striker = st.selectbox('Select Striker', df['batsman'].unique())
bowler = st.selectbox('Select Bowler', df['bowler'].unique())

if st.button('Predict Score'):
    input_data = [
        venue_encoder.transform([venue])[0],
        batting_team_encoder.transform([batting_team])[0],
        bowling_team_encoder.transform([bowling_team])[0],
        striker_encoder.transform([striker])[0],
        bowler_encoder.transform([bowler])[0]
    ]
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    predicted_score = int(model.predict(input_data)[0, 0])
    st.success(f'Predicted Score: {predicted_score}') 