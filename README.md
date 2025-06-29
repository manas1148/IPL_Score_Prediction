# IPL Score Prediction

This repository contains a machine learning project for predicting IPL (Indian Premier League) cricket scores based on historical data. The goal of this project is to build a regression model that can accurately predict the total score of a cricket team based on various features such as venue, batting team, bowling team, batsmen, and bowlers.

![image](res/2.png)

## 🚀 Live Demo

**Try the IPL Score Predictor online:**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_DEPLOY_LINK_HERE)

**Deploy Link:** [https://iplscoreprediction-abwejkv7mvhyegsuzbwoam.streamlit.app/]

*Replace `YOUR_DEPLOY_LINK_HERE` with your actual Streamlit Cloud deployment URL*

## Project Overview

In this project, we have used a dataset ([ipl_data.csv](ipl_data.csv)) containing various features related to IPL matches, including:

- Venue
- Batting Team
- Bowling Team
- Batsmen
- Bowlers
- Runs scored
- Wickets taken
- Overs played
- Runs and wickets in the last 5 overs

The main steps involved in this project are:

1. **Data Preprocessing**:
   - Cleaning and preparing the dataset.
   - Encoding categorical variables using LabelEncoder.
   - Scaling numerical features using MinMaxScaler.

2. **Model Building**:
   - Utilizing Keras (with TensorFlow backend) to build a neural network model.
   - Experimenting with different architectures (number of layers, units per layer) to optimize performance.
   - Training the model using historical IPL data.

3. **Evaluation**:
   - Assessing the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) score.
   - Iteratively refining the model to achieve better predictive accuracy.

4. **Prediction**:
   - Implementing prediction functionality using the trained model.
   - Creating an interactive prediction interface using Streamlit for selecting venue, batting team, bowling team, batsman, and bowler.

5. **Comparison with Other Models**:
   - Implementing Gradient Boosting Regressor and Support Vector Regressor from scikit-learn for benchmarking and comparing performance.

## Results

After experimenting with various neural network architectures and model configurations, the final model achieved the following performance metrics on the test dataset:

- Mean Absolute Error (MAE): 11.95
- Mean Squared Error (MSE): 344.24
- R-squared (R2) Score: 0.592

These metrics indicate that the model is able to predict IPL scores with a high degree of accuracy, explaining approximately 59.2% of the variance in the data.

![image](res/1.png)

## Repository Structure

- `ipl_data.csv`: Dataset used for training and evaluation.
- `IPLScore.ipynb`: Jupyter notebook containing the entire project code with explanations.
- `streamlit_app.py`: Interactive web application built with Streamlit.
- `requirements.txt`: List of Python dependencies required to run the project.
- `.streamlit/config.toml`: Streamlit configuration for deployment.
- `Procfile`: Configuration file for Heroku deployment.
- `runtime.txt`: Python runtime specification for deployment platforms.
- `README.md`: This file, providing an overview of the project, instructions, and results.

## Usage

### Local Development

To run the project locally, follow these steps:

1. Clone the repository:
   ```
   https://github.com/manas1148/IPL_Score_Prediction.git
   cd IPL-score-pred
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

4. Open your browser and go to `http://localhost:8501`

### Jupyter Notebook

You can also run the Jupyter notebook for detailed analysis:

1. Open and run the `IPLScore.ipynb` notebook in Jupyter or any compatible environment.
2. Follow the instructions in the notebook to preprocess the data, build and train the model, and evaluate its performance.

## Deployment

This project can be deployed to various platforms:

### Streamlit Cloud (Recommended)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Connect your repository
4. Set main file path to: `streamlit_app.py`
5. Click "Deploy"

## Future Improvements

- Explore more advanced neural network architectures (e.g., LSTM for sequence data).
- Incorporate additional features such as player statistics, weather conditions, or match history.
- Conduct more thorough hyperparameter tuning and model optimization.
- Add more interactive features to the Streamlit app.
- Implement real-time data updates from live matches.

## Contact


- [GitHub](https://github.com/manas1148)
- [LinkedIn](https://www.linkedin.com/in/manas-rai-kaushik-1b4200242/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Keep Coding 🚀
