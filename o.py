import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

# Load dataset
file_path = "vehicles.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(['id', 'url', 'paint_color', 'image_url', 'posting_date', 'lat', 'VIN',
              'cylinders', 'size', 'drive', 'condition', 'long', 'state', 'county',
              'description', 'region_url'], axis=1)

# Drop rows with missing values in important columns
df = df.dropna(subset=['year', 'odometer', 'manufacturer', 'title_status', 'model', 'fuel', 'transmission', 'type', 'price'])

# Define target and features
y = df['price']
x = df.drop(['price'], axis=1)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Encode categorical features using LabelEncoder
categorical_cols = ['year', 'odometer', 'manufacturer', 'title_status', 'model', 'fuel', 'transmission', 'type']

label_encoders = {}  # Store encoders for later use

for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit & transform training data
    x_train[col] = le.fit_transform(x_train[col])
    
    # Transform test data safely (handling unseen labels)
    x_test[col] = x_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    label_encoders[col] = le  # Save encoder for later use

# Train XGBoost model
xgb = XGBRegressor(enable_categorical=False)  # Disable categorical option as we used Label Encoding
xgb.fit(x_train, y_train)

# Make predictions
y_train_pred = xgb.predict(x_train)
y_test_pred = xgb.predict(x_test)

# Evaluate model performance
train_rmse = root_mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = root_mean_squared_error(y_test, y_test_pred, squared=False)

# Print RMSE values
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
