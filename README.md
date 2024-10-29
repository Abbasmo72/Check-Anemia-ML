# Anemia: Symptoms and Complications
Anemia is a medical condition characterized by a lower-than-normal number of red blood cells or hemoglobin levels in the blood. This condition can arise from various causes, including iron deficiency, vitamin B12 or folate deficiency, genetic disorders, and chronic diseases. Symptoms of anemia may include extreme fatigue, pale skin, dizziness, shortness of breath, and a rapid heartbeat.
The complications of anemia can be serious, especially if left untreated. These complications include a higher risk of infections, cardiovascular problems such as heart failure, and a reduced ability of the body to endure physical activities. In severe cases, anemia can lead to more serious disorders like stroke or heart attack.
Diagnosis of anemia is typically performed through blood tests, and treatment depends on the type of anemia. For iron deficiency, iron supplements and dietary changes may be effective. In the case of vitamin deficiencies, relevant supplements are prescribed.

## Logistic Regression and Its Performance
Logistic regression is a statistical model used to predict the probability of a specific event occurring. This model has widespread applications in fields such as medicine, finance, and social sciences. Unlike linear regression, which is used for predicting numerical values, logistic regression is designed for predicting binary (two-option) variables.
The performance of logistic regression works by using the logistic function to model the probability of an event occurring. This function keeps the predicted values between 0 and 1, making it suitable for predicting probabilities. In this model, the coefficients derived from the data assist in making predictions.
To evaluate the performance of a logistic regression model, metrics such as accuracy, recall, and precision are commonly used. These metrics help us understand how well the model performs and what proportion of predictions are correct.
Logistic regression can provide valuable insights into the importance of each feature in predicting the outcome. This information can be highly beneficial in decision-making and optimizing strategies across various domains.

## Analysis and Prediction of Anemia Using Logistic Regression
This code creates and evaluates a logistic regression model to predict anemia based on blood test data. Here are the steps involved in the code explained in detail:
1. <b>Importing Libraries:</b>
  - pandas: For data manipulation.
  - sklearn.model_selection: To split the data into training and testing sets.
  - sklearn.linear_model: To use the logistic regression model.
  - sklearn.metrics: To calculate the model's accuracy.
2. <b>Loading and Preparing the Dataset:</b>
  - Data is loaded from a CSV file named CBCdata_for_meandeley_csv.csv.
  - Column names are manually set.
  - The first row containing descriptions is removed.
  - Data is converted to numeric types, and any rows with missing values are dropped.
3. <b>Defining Input Features (X) and Target Variable (y):</b>
  - Features include the values from blood tests (RBC, PCV, MCV, MCH, MCHC, RDW, TLC, PLT).
  - The target variable indicates whether hemoglobin (HGB) is less than 12. If it is less, it is marked as 1 (anemia), otherwise as 0 (no anemia).
4. <b>Splitting the Data:</b>
  - The data is split into 80% for training and 20% for testing.
5. <b>Training the Model:</b>
  - A logistic regression model is created and trained on the training data.
6. <b>Evaluating the Model:</b>
  - Predictions are made for the test set, and the accuracy of the model is calculated and printed.
7. <b>Prediction with User Input:</b>
  - The program prompts the user to enter values for the features.
  - A prediction is made based on the user's input, and the result (whether anemia is likely or not) is displayed.





