# User Guide

Visit the Streamlit here: [it3385-mlops-group4.streamlit.app](https://it3385-mlops-group4.streamlit.app)

This document details how to use the application.

## Home Page
The home page is where you can navigate to the respective sections of the app.

The app features three predictors:
- Wheat classifier,
- Melbourne residential housing price predictor,
- Used car prices predictor

## Wheat Classifier
Developed by: Choo Tze Hsuen, 220926F
Predict what type of kernel based on its physical attributes.

![Picture of wheat predictor page interface](/.github/images/wheat_1.webp)
![Picture of wheat predictor page interface](/.github/images/wheat_2.webp)
![Picture of wheat predictor page interface](/.github/images/wheat_3.webp)

####  Features
- Single Prediction: Enter kernel attributes manually to instantly see the predicted wheat variety along with the model’s confidence score.
- Batch Prediction (CSV): Upload a CSV file with multiple kernel records to generate predictions for each row.
- Template CSV Download: Download a ready-made CSV template to try out the batch prediction feature.
- Batch Prediction Summary: View the bar chart visual showing how many samples were classified into each wheat type, and hover over each bar to see the average confidence score.

## Melbourne Residential Housing Predictor
Developed by: Muhammad Aniq Sufi Bin Ismail, 232237W
Predict the resale price of used cars based on its physical and non-physical characteristics.

#### Features
- Input several physical and non-physical features about the car and its origins to return a predicted resale price for the car.
- For individual predictions, view several charts that indicate the expected uncertainty of the models predictive results. The lower the amount (closer to 0) the better the predictive ability (Take note the metrics are scaled down so it's more readable).
- View the error range of the predictive model based on the training metrics (RMSE = 368541.95). NOTE: If error range exceeds below zero, the interpretation can be interpreted as 0 to upper bound.
- Perform batch predictions.
- For batch predictions, users are able to view several charts that tells them about the insights from their batch predictions (E.g., Distribution of predicted car prices, Top locations by average predicted price, Power of vehicle vs predicted price, Predicted price vs Fuel Type).

## Used Car Prices Predictor