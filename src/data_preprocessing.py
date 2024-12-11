import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from haversine import haversine, Unit

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with your dataset path

# Data Cleaning
df = df.dropna()
df = df[(df['fare_amount'] > 0) & (df['trip_duration'] > 0)]

# Feature Engineering
def haversine_distance(row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])
    return haversine(pickup, dropoff)

df['distance'] = df.apply(haversine_distance, axis=1)
df['hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour
df['day_of_week'] = pd.to_datetime(df['pickup_datetime']).dt.dayofweek

# Train-Test Split
features = ['distance', 'hour', 'day_of_week']
target_fare = 'fare_amount'
target_duration = 'trip_duration'

X = df[features]
y_fare = df[target_fare]
y_duration = df[target_duration]

X_train_fare, X_test_fare, y_train_fare, y_test_fare = train_test_split(X, y_fare, test_size=0.2, random_state=42)
X_train_duration, X_test_duration, y_train_duration, y_test_duration = train_test_split(X, y_duration, test_size=0.2, random_state=42)
