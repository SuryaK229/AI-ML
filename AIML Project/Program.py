import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset (replace the path with the actual path to your file)
data = pd.read_csv('Dataset location')

# Drop the unnamed index column
data = data.drop(columns=['Unnamed: 0'])

# List of all symptoms in the dataset (excluding the 'diseases' column)
symptom_columns = data.columns[1:]

# Features (symptoms) and target (diseases)
X = data[symptom_columns]
y = data['diseases']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Symptom checker function
def check_symptoms(user_symptoms):
    # Create a binary vector for the input symptoms
    user_symptoms_vector = [1 if symptom in user_symptoms else 0 for symptom in symptom_columns]
    
    # Predict the disease based on the input symptoms
    predicted_disease = model.predict([user_symptoms_vector])
    return predicted_disease[0]

# Manual input of symptoms
print("Please enter your symptoms separated by commas.")
user_input = input("Example: 'fever, cough, headache'\n")

# Process the user input and split into a list of symptoms
user_symptoms = user_input.split(',')

# Remove any extra spaces around symptoms
user_symptoms = [symptom.strip() for symptom in user_symptoms]

# Predict the disease
predicted_disease = check_symptoms(user_symptoms)
print(f"The predicted disease based on your symptoms is: {predicted_disease}")
