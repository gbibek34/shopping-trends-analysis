import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Loading dataset
dataset_path = 'shopping_behavior_updated.csv'
df = pd.read_csv(dataset_path)

# Selection of relevant features: age, gender, and location
features = ['Age', 'Gender', 'Location']
target_item = 'Item Purchased'

# Encoding categorical features (gender and location)
le_gender = LabelEncoder()
le_location = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Location'] = le_location.fit_transform(df['Location'])

# Separation of numeric and categorical features
numeric_features = ['Age']
categorical_features = ['Gender', 'Location']

# Handle missing values for numeric features
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

# Split data into training and testing sets
X = df[features]
y_item = df[target_item]
X_train, X_test, y_item_train, y_item_test = train_test_split(X, y_item, test_size=0.2, random_state=42)

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

# Initialize KNN classifier
knn_item = KNeighborsClassifier()

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7,9,11],  # Trying   different values of N
    'weights': ['uniform', 'distance'], #trying 2 different weight techniques
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'] #trying out the possible distance formula
}

# Performing the grid search
grid_search = GridSearchCV(knn_item, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_item_train)

# Getting the best parameters for optimal performance of modal 
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the model with best parameters
best_knn_item = KNeighborsClassifier(**best_params)
best_knn_item.fit(X_train_scaled, y_item_train)

# Make predictions
predicted_item = best_knn_item.predict(X_test_scaled)
accuracy = (predicted_item == y_item_test).mean()
print(f"Accuracy on test set: {accuracy:.2f}")
