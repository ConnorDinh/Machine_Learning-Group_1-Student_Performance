from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd


class KNN_Regression:
    def __new__(self):
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
        y_train = np.load("y_train.npy")
        y_test = np.load("y_test.npy")

        # KNN uses distance, so scaling is important
        self.knn = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5)
        )

        # Cross-validation on training data
        cv_scores = cross_val_score(
            self.knn,
            X_train,
            y_train,
            cv=5,
            scoring="r2"
        )

        # Train the model
        self.knn.fit(X_train, y_train)

        # Make predictions
        knn_pred = self.knn.predict(X_test)

        # Evaluate the model
        knn_mae = mean_absolute_error(y_test, knn_pred)

        knn_rmse = np.sqrt(
            mean_squared_error(y_test, knn_pred)
        )

        knn_r2 = r2_score(y_test, knn_pred)

        # KNN does not have coefficients or feature_importances_
        # so this is just used to keep output format similar
        features = pd.DataFrame({
            "Feature": [
                "weekly_self_study_hours",
                "attendance_percentage",
                "class_participation"
            ],
            "Importance": [
                "N/A",
                "N/A",
                "N/A"
            ]
        })

        return knn_mae, knn_rmse, knn_r2, features