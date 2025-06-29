import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# Define and train the model using Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# Streamlit UI
st.title('IPL Score Predictor')
st.markdown("---")

# Add some styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏏 IPL Score Predictor</h1>', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏟️ Match Details")
    venue = st.selectbox('Select Venue', sorted(df['venue'].unique()))
    batting_team = st.selectbox('Select Batting Team', sorted(df['bat_team'].unique()))
    bowling_team = st.selectbox('Select Bowling Team', sorted(df['bowl_team'].unique()))

with col2:
    st.subheader("👥 Player Details")
    striker = st.selectbox('Select Striker', sorted(df['batsman'].unique()))
    bowler = st.selectbox('Select Bowler', sorted(df['bowler'].unique()))

st.markdown("---")

if st.button('🎯 Predict Score', type="primary"):
    with st.spinner('Training model and making prediction...'):
        # Prepare input data
        input_data = [
            venue_encoder.transform([venue])[0],
            batting_team_encoder.transform([batting_team])[0],
            bowling_team_encoder.transform([bowling_team])[0],
            striker_encoder.transform([striker])[0],
            bowler_encoder.transform([bowler])[0]
        ]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)
        
        # Make prediction
        predicted_score = int(model.predict(input_data)[0])
        
        # Display result with styling
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🎯 Predicted Score: <span style="color: #1f77b4; font-size: 2rem;">{predicted_score}</span></h3>
            <p><strong>Venue:</strong> {venue}</p>
            <p><strong>Batting Team:</strong> {batting_team}</p>
            <p><strong>Bowling Team:</strong> {bowling_team}</p>
            <p><strong>Striker:</strong> {striker}</p>
            <p><strong>Bowler:</strong> {bowler}</p>
        </div>
        """, unsafe_allow_html=True)

# Add some information about the model
st.markdown("---")
st.subheader("📊 Model Information")
st.markdown("""
- **Model Type:** Random Forest Regressor
- **Features:** Venue, Teams, Players
- **Training Data:** Historical IPL match data
- **Accuracy:** Optimized for score prediction
""")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit and Machine Learning</p>
</div>
""", unsafe_allow_html=True) 