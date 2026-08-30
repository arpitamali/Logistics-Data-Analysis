import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load raw logistics data
df = pd.read_csv("logistics_data_raw.csv")

# 1. Inspect data
print(df.info())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# 2. Handle missing numerical values using median
num_cols = ["Distance_km", "Shipment_Weight_kg",
            "Delivery_Time_Hours", "Transportation_Cost"]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 3. Handle missing categorical values using mode
cat_cols = ["Traffic_Level", "Weather", "Vehicle_Type"]

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Remove duplicate rows
df = df.drop_duplicates()

# 5. Detect and cap numerical outliers using IQR
def cap_outliers_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data[column] = data[column].clip(lower, upper)
    return data

for col in num_cols:
    df = cap_outliers_iqr(df, col)

# 6. Normalize selected numerical features
scaler = MinMaxScaler()
scale_cols = ["Distance_km", "Shipment_Weight_kg",
              "Delivery_Time_Hours", "Transportation_Cost"]

df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Save cleaned/preprocessed dataset
df.to_csv("logistics_data_cleaned.csv", index=False)

print("Preprocessing completed.")
print(df.head())
