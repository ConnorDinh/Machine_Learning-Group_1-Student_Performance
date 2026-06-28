from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

class Random_Forest_Regression:
	def __new__(self):
		X_train = np.load("X_train.npy")
		X_test = np.load("X_test.npy")
		y_train = np.load("y_train.npy")
		y_test = np.load("y_test.npy")

		self.rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
		cv_scores = cross_val_score(self.rf, X_train, y_train, cv=5, scoring='r2')
		self.rf.fit(X_train, y_train)

		'''print("R² scores:", cv_scores)
		print("Mean R²:", np.mean(cv_scores))
		print("Std:", np.std(cv_scores))'''

		rf_pred = self.rf.predict(X_test)

		rf_mae = mean_absolute_error(y_test, rf_pred)

		rf_rmse = np.sqrt(
		    mean_squared_error(y_test, rf_pred)
		)

		rf_r2 = r2_score(y_test, rf_pred)
		features = pd.DataFrame({'Feature': ['weekly_self_study_hours', 'attendance_percentage', 'class_participation'], 'Importance': self.rf.feature_importances_})

		'''plt.figure(figsize=(18,8))

		plot_tree(
			self.rf.estimators_[0],
			filled=True,
			rounded=True,
			fontsize=10
		)

		plt.show()'''

		return rf_mae, rf_rmse, rf_r2, features.sort_values(by='Importance', ascending=False)