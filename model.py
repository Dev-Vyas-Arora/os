# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Read Excel file
df = pd.read_excel("delivery_priority_data.xlsx")

# Step 2: Encode categorical columns
le_type = LabelEncoder()
le_priority = LabelEncoder()

df['Delivery_Type'] = le_type.fit_transform(df['Delivery_Type'])
df['Priority'] = le_priority.fit_transform(df['Priority']) # Target variable

# Step 3: Split features and labels
X = df.drop(['Product_ID', 'Priority'], axis=1)
y = df['Priority']

# Step 4: Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train Decision Tree (Gini Index)
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Visualize Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns, class_names=['High','Medium','Low'], filled=True, rounded=True)
plt.show()


import joblib

# Save model
joblib.dump(model, 'delivery_priority_model.pkl')

# Save encoders and scaler too
joblib.dump(le_type, 'delivery_type_encoder.pkl')
joblib.dump(le_priority, 'priority_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and encoders saved successfully!")