import pandas as pd
import matplotlib.pyplot as plt

class DataExtracting:
    def __init__(self):
        #Data loading
        self.df = pd.read_csv("student_performance.csv")

    def get_data_frame(self):
        return self.df

    def isMissingValue(self):
        return self.df.isnull().sum()

    def get_describe(self):
        return self.df.describe()

    def get_relationship(self):
        sample_df = self.df.sample(5000, random_state=42)

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

        #Relationship between class participation and score
        plt.scatter(
            sample_df['student_id'],
            sample_df['total_score']
        )
        plt.xlabel("Student ID")
        plt.ylabel("Total Score")
        plt.title("Student ID vs Total Score")
        plt.show()
