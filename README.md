# Biochar-Absorption-Predictor

1. Overview
This project provides a user-friendly Graphical User Interface (GUI) application for predicting the uranium adsorption capacity of acid-modified biochar. Developed in Python, it integrates a pre-trained and optimized CatBoost regression model to offer instant predictions and, crucially, quantile-based confidence intervals​ to quantify prediction uncertainty.
Our tool is designed to bridge the gap between complex data-driven modeling and practical experimental workflows. By transforming a sophisticated machine learning model into an accessible desktop application, we empower researchers and practitioners in environmental science and materials engineering to intelligently guide their experiments, optimize material synthesis, and contribute to more efficient radioactive wastewater treatment strategies without requiring any programming expertise.

2. Key Features
Instant Predictions:​ Enter 15 key physicochemical feature parameters and receive immediate adsorption capacity predictions.
Robust Uncertainty Quantification:​ Moves beyond simple point estimates by reporting a 95% confidence interval​ calculated using quantile regression. This interval dynamically reflects the model's confidence, widening for inputs in unfamiliar regions.
Guided Input Ranges:​ Input fields are accompanied by tooltips suggesting valid ranges based on the training data distribution:
Compact Support:​ Suggested range matches the observed min/max values.
Long-Tailed Features:​ Suggested range uses the 5th-95th percentile to avoid extrapolation into sparse data regions.
Flexible yet Safe:​ Allows for limited extrapolation to support exploratory research while explicitly warning users against extensive extrapolation that can lead to unreliable results.
No Coding Required:​ Fully encapsulates the machine learning pipeline, making advanced predictive capabilities accessible to all lab members.

3. Usage
3.1. Making a Prediction
The application window will open, divided into three logical input modules.
Enter the values for the 15 feature parameters into the corresponding fields.
Tip:​ Hover your mouse over an input field to see a tooltip with the recommended value range and justification.
Click the "Predict"​ button.
The results panel will display:
Predicted Adsorption Capacity:​ The median predicted value.
95% Confidence Interval:​ The lower and upper bounds (e.g., [Lower_Bound, Upper_Bound]), indicating the range where the true value is expected to lie with 95% probability.
3.2. Interpreting the Results & Confidence Interval
The confidence interval is the most critical part of the output.
A narrow interval​ suggests the input conditions are well-represented in the training data, and the model is highly confident.
A wide interval​ indicates the input is in a sparsely populated region of the feature space. The model's prediction is less certain, and the result should be treated as a rough estimate. This is a direct warning against high uncertainty.

4. Contributing
Contributions to improve the tool are welcome. For major changes, please open an issue first to discuss what you would like to change.
