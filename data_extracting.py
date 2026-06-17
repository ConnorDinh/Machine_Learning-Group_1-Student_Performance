import pandas as pd
import matplotlib.pyplot as plt

#Data loading
df = pd.read_csv("student_performance.csv")

print(df.head())
print(df.shape)
print(df.info())

#Value missing checking
print(df.isnull().sum())


print(df.describe())

sample_df = df.sample(5000, random_state=42)

#Relationship between study hours and score
plt.scatter(
    sample_df['weekly_self_study_hours'],
    sample_df['total_score']
)
plt.xlabel("Study Hours")
plt.ylabel("Total Score")
plt.title("Study Hours vs Total Score")
plt.show()

#Relationship between attendance percentage and score
plt.scatter(
    sample_df['attendance_percentage'],
    sample_df['total_score']
)
plt.xlabel("Attendance")
plt.ylabel("Total Score")
plt.title("Attendance vs Total Score")
plt.show()

#Relationship between class participation and score
plt.scatter(
    sample_df['class_participation'],
    sample_df['total_score']
)
plt.xlabel("Class Participation")
plt.ylabel("Total Score")
plt.title("Participation vs Total Score")
plt.show()

