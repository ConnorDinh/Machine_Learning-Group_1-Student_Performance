import numpy as np
import pandas as pd
from Linear_Regression import Linear_Regression
from Decision_Tree import Decision_Tree
from Random_Forest_Regression import Random_Forest_Regression
import matplotlib.pyplot as plt

lr_mae, lr_rmse, lr_r2, lr_coef = Linear_Regression()
dt_mae, dt_rmse, dt_r2, dt_features = Decision_Tree()
rf_mae, rf_rmse, rf_r2, rf_features = Random_Forest_Regression()

results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree',
        'Random Forest'
    ],
    'MAE': [
        lr_mae,
        dt_mae,
        rf_mae
    ],
    'RMSE': [
        lr_rmse,
        dt_rmse,
        rf_rmse
    ],
    'R²': [
        lr_r2,
        dt_r2,
        rf_r2
    ]
})

print(results)

results.set_index('Model')[['MAE', 'RMSE', 'R²']].plot(kind='bar', figsize=(8, 5))

plt.title('Comparison of Regression Models')
plt.ylabel('Metric Value')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Linear Regression:")
print(lr_coef)

print("\nDecision Tree:")
print(dt_features)

print("\nRandom Forest Regression:")
print(rf_features)

'''
plt.figure(figsize=(7,4))

plt.bar(
    rf_features['Feature'],
    rf_features['Importance']
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()'''
