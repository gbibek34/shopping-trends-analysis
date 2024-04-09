import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class ShoppingBehaviorModel:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)

        # Encoding categorical features
        self.le_gender = LabelEncoder()
        self.le_payment = LabelEncoder()
        self.df['Gender'] = self.le_gender.fit_transform(self.df['Gender'])
        self.df['Payment Method'] = self.le_payment.fit_transform(self.df['Payment Method'])

        # Handling missing values for numeric features
        self.numeric_features = ['Age']
        self.df[self.numeric_features] = self.df[self.numeric_features].fillna(self.df[self.numeric_features].mean())

        # Split data into training and testing sets
        features = ['Age', 'Gender', 'Location']
        target_item = 'Item Purchased'
        target_payment = 'Payment Method'
        X = self.df[features]
        y_item = self.df[target_item]
        y_payment = self.df[target_payment]
        self.X_train, self.X_test, self.y_item_train, self.y_item_test, self.y_payment_train, self.y_payment_test = \
            train_test_split(X, y_item, y_payment, test_size=0.2, random_state=42)

        # Standardize numeric features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train[self.numeric_features])
        self.X_test_scaled = self.scaler.transform(self.X_test[self.numeric_features])

        # Initializing KNN classifiers
        self.knn_item = KNeighborsClassifier(n_neighbors=3,metric='euclidean',weights='uniform')
        self.knn_payment = KNeighborsClassifier(n_neighbors=3, metric='euclidean',weights='uniform')

        # Training the models
        self.knn_item.fit(self.X_train_scaled, self.y_item_train)
        self.knn_payment.fit(self.X_train_scaled, self.y_payment_train)

    def predict(self, user_age, user_gender, user_location):
        user_gender_encoded = self.le_gender.transform([user_gender])[0]

        # Find encoded location
        user_location_encoded = self.df.loc[self.df['Location'] == user_location, 'Location'].iloc[0]

        new_customer = pd.DataFrame({
            'Age': [user_age],
            'Gender': [user_gender_encoded],
            'Location': [user_location_encoded]
        })

        new_customer_scaled = self.scaler.transform(new_customer[self.numeric_features])

        predicted_item = self.knn_item.predict(new_customer_scaled)[0]
        predicted_payment = self.knn_payment.predict(new_customer_scaled)[0]

        predicted_payment_decoded = self.le_payment.inverse_transform([predicted_payment])[0]

        return predicted_item, predicted_payment_decoded
