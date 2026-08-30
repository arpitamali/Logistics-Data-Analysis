import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logistics_data_week3.csv")

# Basic EDA
print(df.info())
print(df.describe())

# Central tendency
print("Mean delivery time:", df["Delivery_Time_Hours"].mean())
print("Median delivery time:", df["Delivery_Time_Hours"].median())
print("Mean transportation cost:", df["Transportation_Cost"].mean())

# Correlation
print(df[["Distance_km", "Shipment_Weight_kg",
          "Delivery_Time_Hours", "Transportation_Cost"]].corr())

# Visualization 1: delivery-time distribution
plt.figure(figsize=(7,5))
plt.hist(df["Delivery_Time_Hours"], bins=15)
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Shipments")
plt.title("Distribution of Delivery Time")
plt.show()

# Visualization 2: distance vs delivery time
plt.figure(figsize=(7,5))
plt.scatter(df["Distance_km"], df["Delivery_Time_Hours"], alpha=0.75)
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hours)")
plt.title("Distance vs Delivery Time")
plt.show()

# Visualization 3: traffic vs delivery time
traffic_avg = df.groupby("Traffic_Level")["Delivery_Time_Hours"].mean()
traffic_avg = traffic_avg.reindex(["Low", "Medium", "High"])
traffic_avg.plot(kind="bar")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (hours)")
plt.title("Average Delivery Time by Traffic Level")
plt.xticks(rotation=0)
plt.show()

# Visualization 4: vehicle type vs cost
vehicle_avg = df.groupby("Vehicle_Type")["Transportation_Cost"].mean()
vehicle_avg.sort_values().plot(kind="bar")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Transportation Cost")
plt.title("Average Transportation Cost by Vehicle Type")
plt.xticks(rotation=0)
plt.show()
