import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# Load Dataset
# ==============================
print("Loading dataset...")
df = pd.read_csv("random.csv")

print("\nFirst 5 Rows:")
print(df.head())

# ==============================
# Check Missing Values
# ==============================
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# ==============================
# Encode Categorical Columns
# ==============================
categorical_columns = [
    "Brand",
    "Fuel",
    "Transmission",
    "Owner"
]

label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

print("\nCategorical Encoding Completed.")

# ==============================
# Define Features & Target
# ==============================
x = df.drop("Price", axis=1)
y = df["Price"]

# ==============================
# Split Dataset
# ==============================
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(x_train))
print("Testing Samples  :", len(x_test))

# ==============================
# Train Model
# ==============================
print("\nTraining Model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(x_train, y_train)

print("Training Completed!")

# ==============================
# Prediction
# ==============================
y_pred = model.predict(x_test)

# ==============================
# Model Evaluation
# ==============================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== Model Performance ==========")
print(f"MAE       : {mae:.2f}")
print(f"MSE       : {mse:.2f}")
print(f"RMSE      : {rmse:.2f}")
print(f"R2 Score  : {r2:.4f}")

# ==============================
# Save Model
# ==============================
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved: car_price_model.pkl")

# ==============================
# Save Label Encoders
# ==============================
with open("label_encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)

print("Label Encoders Saved: label_encoders.pkl")

# ==============================
# Feature Importance
# ==============================
importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== Feature Importance ==========")
print(importance)

# ==============================
# Training Completed
# ==============================
print("\n===================================")
print("Training Completed Successfully!")
print("===================================")