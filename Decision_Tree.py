from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

class Decision_Tree:
	def __new__(self):
		X_train = np.load("X_train.npy")
		X_test = np.load("X_test.npy")
		y_train = np.load("y_train.npy")
		y_test = np.load("y_test.npy")

		self.dt = DecisionTreeRegressor(max_depth=10, random_state=42)
		cv_scores = cross_val_score(self.dt, X_train, y_train, cv=5, scoring='r2')
		self.dt.fit(X_train, y_train)

		'''print("R² scores:", cv_scores)
		print("Mean R²:", np.mean(cv_scores))
		print("Std:", np.std(cv_scores))'''

		dt_pred = self.dt.predict(X_test)

		dt_mae = mean_absolute_error(y_test, dt_pred)

		dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

		dt_r2 = r2_score(y_test, dt_pred)
		features = pd.DataFrame({'Feature': ['weekly_self_study_hours', 'attendance_percentage', 'class_participation'], 'Importance': self.dt.feature_importances_})

		'''plt.figure(figsize=(18,8))

		plot_tree(
			self.dt,
			filled=True,
			rounded=True,
			fontsize=10
		)

		plt.show()'''

		return dt_mae, dt_rmse, dt_r2, features.sort_values(by='Importance', ascending=False)