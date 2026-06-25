from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import pandas as pd

class Linear_Regression:
	def __new__(self):
		X_train = np.load("X_train.npy")
		X_test = np.load("X_test.npy")
		y_train = np.load("y_train.npy")
		y_test = np.load("y_test.npy")

		self.lr = LinearRegression()
		cv_scores = cross_val_score(self.lr, X_train, y_train, cv=5, scoring='r2')

		self.lr.fit(X_train, y_train)

		'''print("R² scores:", cv_scores)
		print("Mean R²:", np.mean(cv_scores))
		print("Std:", np.std(cv_scores))

		print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))'''
		lr_pred = self.lr.predict(X_test)

		lr_mae = mean_absolute_error(y_test, lr_pred)

		lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

		lr_r2 = r2_score(y_test, lr_pred)

		return lr_mae, lr_rmse, lr_r2, pd.DataFrame({'Feature': ['weekly_self_study_hours', 'attendance_percentage', 'class_participation'], 'Coefficient': self.lr.coef_})