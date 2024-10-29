<div align="center">

## تشخیص کم خونی
<img src="https://i.pinimg.com/originals/a8/f2/84/a8f284cf430e9b5ee5df1d02395ba11c.gif" height="200px" width="500px">
</div>
<hr>

### کم خونی: علائم و عوارض
کم خونی یک وضعیت پزشکی است که با تعداد کمتر از حد طبیعی گلبول های قرمز یا سطح هموگلوبین در خون مشخص می شود. این عارضه می تواند به دلایل مختلفی از جمله کمبود آهن، کمبود ویتامین B12 یا فولات، اختلالات ژنتیکی و بیماری های مزمن ایجاد شود. علائم کم خونی ممکن است شامل خستگی شدید، رنگ پریدگی پوست، سرگیجه، تنگی نفس و ضربان قلب سریع باشد.
عوارض کم خونی می تواند جدی باشد، به خصوص اگر درمان نشود. این عوارض شامل خطر بیشتر عفونت ها، مشکلات قلبی عروقی مانند نارسایی قلبی و کاهش توانایی بدن برای تحمل فعالیت های فیزیکی است. در موارد شدید، کم خونی می تواند منجر به اختلالات جدی تری مانند سکته مغزی یا حمله قلبی شود.
تشخیص کم خونی معمولاً از طریق آزمایش خون انجام می شود و درمان بستگی به نوع کم خونی دارد. برای کمبود آهن، مکمل های آهن و تغییرات رژیم غذایی ممکن است موثر باشد. در صورت کمبود ویتامین، مکمل های مربوطه تجویز می شود.

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

## Here’s a line-by-line analysis of the provided code:
1. This line imports the pandas library, which is used for data manipulation and analysis.
```python
import pandas as pd
```
2. This line imports the train_test_split function from the sklearn.model_selection module, which is used to split the dataset into training and testing sets.
```python
from sklearn.model_selection import train_test_split
```
3. This line imports the LogisticRegression model from the sklearn.linear_model module, which will be used to create the logistic regression model.
```python
from sklearn.linear_model import LogisticRegression
```
4. This line imports the accuracy_score function from the sklearn.metrics module, which is used to calculate the accuracy of the model.
```python
from sklearn.metrics import accuracy_score
```
5. This line loads the data from a CSV file named CBCdata_for_meandeley_csv.csv and stores it in a DataFrame called data.
```python
data = pd.read_csv('CBCdata_for_meandeley_csv.csv')
```
6. This line manually sets the column names of the DataFrame.
```python
data.columns = ['S.No', 'Age', 'Sex', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT', 'HGB']
```
7. This line removes the first row of the DataFrame, which usually contains descriptions or headers.
```python
data = data[1:]
```
8. This line attempts to convert all values in the DataFrame to numeric types. If any errors occur (such as non-numeric values), those values are converted to NaN.
```python
data = data.apply(pd.to_numeric, errors='coerce')
```
9. This line drops all rows containing missing values (NaN) from the DataFrame.
```python
data = data.dropna()
```
10. This line defines the input features (X), which include the values from various blood tests.
```python
X = data[['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT']]
```
11. This line defines the target variable (y). It sets the value to 1 (indicating anemia) if the hemoglobin (HGB) level is less than 12, and 0 (indicating no anemia) otherwise.
```python
y = (data['HGB'] < 12).astype(int)
```
12. This line splits the data into training and testing sets. It allocates 80% of the data for training and 20% for testing. The random_state=42 ensures that the data is split the same way every time the code is run.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
13. This line creates a new instance of the logistic regression model.
```python
model = LogisticRegression()
```
14. This line trains the logistic regression model using the training data.
```python
model.fit(X_train, y_train)
```
15. This line makes predictions on the test set and stores the results in y_pred.
```python
y_pred = model.predict(X_test)
```
16. This line calculates the accuracy of the model by comparing the predicted values to the actual values.
```python
accuracy = accuracy_score(y_test, y_pred)
```
17. This line prints the accuracy of the model formatted to two decimal places.
```python
print(f"Model accuracy: {accuracy:.2f}")
```
18. This line prompts the user to input values for the features and stores these values in a list called user_input. Each input is converted to a float.
```python
user_input = [float(input(f"Please enter the value for {col}: ")) for col in X.columns]
```
19. This line makes a prediction based on the user's input and stores the result in prediction.
```python
prediction = model.predict([user_input])
```
20. This section checks the prediction result and displays an appropriate message to the user: if the prediction is 1, it indicates that anemia is likely; if it is 0, it indicates no anemia.
```python
if prediction[0] == 1:
    print("Result: Anemia is likely.")
else:
    print("Result: No anemia.")
```

## Python Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and prepare the dataset
data = pd.read_csv('CBCdata_for_meandeley_csv.csv')
data.columns = ['S.No', 'Age', 'Sex', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT', 'HGB']

# Remove the first row containing descriptions
data = data[1:]

# Convert data to numeric and drop any rows with missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Define input features (X) and target variable (y)
X = data[['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT']]
y = (data['HGB'] < 12).astype(int)  # Anemia is indicated if Hemoglobin is less than 12

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Prediction with new user input
user_input = [float(input(f"Please enter the value for {col}: ")) for col in X.columns]
prediction = model.predict([user_input])

# Display result based on prediction
if prediction[0] == 1:
    print("Result: Anemia is likely.")
else:
    print("Result: No anemia.")
```
