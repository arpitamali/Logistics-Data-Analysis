import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logistics_data.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

average_delivery_time = df["Delivery_Time_Hours"].mean()
delay_rate = df["Delayed"].mean() * 100

print("Average Delivery Time:", average_delivery_time)
print("Delay Rate:", delay_rate)

df["Delivery_Time_Hours"].plot(kind="hist", bins=20)
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Shipments")
plt.title("Distribution of Delivery Time")
plt.show()
