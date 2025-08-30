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
Developed by: Choo Tze Hsuen, 220926F | Predict what type of kernel based on its physical attributes.

![Picture of wheat predictor page interface](/.github/images/wheat_1.webp)
![Picture of wheat predictor page interface](/.github/images/wheat_2.webp)
![Picture of wheat predictor page interface](/.github/images/wheat_3.webp)

####  Features
- Single Prediction: Enter kernel attributes manually to instantly see the predicted wheat variety along with the model’s confidence score.
- Batch Prediction (CSV): Upload a CSV file with multiple kernel records to generate predictions for each row.
- Template CSV Download: Download a ready-made CSV template to try out the batch prediction feature.
- Batch Prediction Summary: View the bar chart visual showing how many samples were classified into each wheat type, and hover over each bar to see the average confidence score.

## Melbourne Residential Housing Predictor
Developed by: Ian Chia Bing Jun, 230746D | Predict the selling price of residential properties in Melbourne based on their physical characteristics and location attributes.

![Picture of housing predictor page interface](/.github/images/melbourne_1.jpeg)
![Picture of housing predictor page interface](/.github/images/melbourne_2.jpeg)

#### Features
- Input several physical features of the property such as number of rooms, bathrooms, car spaces, building area, land size, and year built.
- Input location-based attributes such as suburb, region, and distance from the Central Business District (CBD).
- For individual predictions, view charts showing the model’s performance, including error distribution and residual plots, to better understand the accuracy of the prediction.
- View the error range of the predictive model based on training metrics (e.g., RMSE ≈ 323,945). If the lower bound of the error range falls below zero, the predicted price range should be interpreted as starting from zero.
- Batch predictions can be performed by uploading multiple property records at once.
- When doing batch predictions, users can explore insights such as the distribution of predicted house prices, top suburbs or regions by average predicted price, and the relationship between key features (e.g., land size, distance to CBD) and predicted price.

## Used Car Prices Predictor
Developed by: Muhammad Aniq Sufi Bin Ismail, 232237W | Predict the resale price of used cars based on its physical and non-physical characteristics.

![Picture of car prices predictor page interface](/.github/images/car_1.jpeg)
![Picture of car prices predictor page interface](/.github/images/car_2.jpeg)

#### Features
- Input several physical and non-physical features about the car and its origins to return a predicted resale price for the car.
- For individual predictions, view several charts that indicate the expected uncertainty of the models predictive results. The lower the amount (closer to 0) the better the predictive ability (Take note the metrics are scaled down so it's more readable).
- View the error range of the predictive model based on the training metrics (RMSE = 368541.95). NOTE: If error range exceeds below zero, the interpretation can be interpreted as 0 to upper bound.
- Perform batch predictions.
- For batch predictions, users are able to view several charts that tells them about the insights from their batch predictions (E.g., Distribution of predicted car prices, Top locations by average predicted price, Power of vehicle vs predicted price, Predicted price vs Fuel Type).