# Machine_Learning-Group_1-Student_Performance
# Student Performance Prediction

## Project Overview

This project uses machine learning regression models to predict a student's **total score** based on academic behavior factors such as weekly self-study hours, attendance percentage, and class participation.

The goal is to compare different regression models and determine which one predicts student performance most accurately.

## Dataset

The dataset used in this project is `student_performance.csv`.

The main variables used are:

- `weekly_self_study_hours` — number of hours a student studies independently each week
- `attendance_percentage` — student attendance percentage
- `class_participation` — student class participation score
- `total_score` — final score being predicted

Although `student_id` is included in the dataset and visualized during exploration, it is not used as a prediction feature because it is only an identifier and does not directly explain academic performance.

## Project Workflow

### 1. Data Loading and Exploration

The dataset is loaded using pandas. Basic checks are performed to look for missing values and summarize the data. Scatterplots are also created to visualize relationships between the predictors and total score. :contentReference[oaicite:0]{index=0}

The visualizations include:

- Study Hours vs Total Score
- Attendance vs Total Score
- Class Participation vs Total Score
- Student ID vs Total Score

From the plots, weekly self-study hours appears to have the strongest visible relationship with total score.

### 2. Data Splitting

The features used for prediction are:

python
weekly_self_study_hours
attendance_percentage
class_participation
