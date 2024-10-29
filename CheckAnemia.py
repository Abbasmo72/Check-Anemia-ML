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
