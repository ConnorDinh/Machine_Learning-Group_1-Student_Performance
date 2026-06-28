from data_extracting import DataExtracting
from sklearn.model_selection import train_test_split
import numpy as np

de = DataExtracting()
df = de.get_data_frame()

print(de.isMissingValue())

X = df[['weekly_self_study_hours', 'attendance_percentage', 'class_participation']].values
y = df['total_score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
np.save("X_train", X_train)
np.save("X_test", X_test)
np.save("y_train", y_train)
np.save("y_test", y_test)

