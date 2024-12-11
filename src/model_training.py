# Train Fare Prediction Model
fare_model = RandomForestRegressor(random_state=42)
fare_model.fit(X_train_fare, y_train_fare)
fare_predictions = fare_model.predict(X_test_fare)
print("Fare Model RMSE:", np.sqrt(mean_squared_error(y_test_fare, fare_predictions)))

# Train Duration Prediction Model
duration_model = RandomForestRegressor(random_state=42)
duration_model.fit(X_train_duration, y_train_duration)
duration_predictions = duration_model.predict(X_test_duration)
print("Duration Model RMSE:", np.sqrt(mean_squared_error(y_test_duration, duration_predictions)))
