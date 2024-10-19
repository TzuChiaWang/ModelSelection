import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define the data
true_class = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
prob_m1 = np.array([0.50, 0.69, 0.44, 0.55, 0.67, 0.47, 0.08, 0.15, 0.45, 0.35])
prob_m2 = np.array([0.61, 0.03, 0.68, 0.31, 0.45, 0.09, 0.38, 0.05, 0.01, 0.04])

# Calculate ROC curve and AUC for M1
fpr_m1, tpr_m1, _ = roc_curve(true_class, prob_m1)
roc_auc_m1 = auc(fpr_m1, tpr_m1)

# Calculate ROC curve and AUC for M2
fpr_m2, tpr_m2, _ = roc_curve(true_class, prob_m2)
roc_auc_m2 = auc(fpr_m2, tpr_m2)

# Plot ROC curves
plt.figure()
plt.plot(fpr_m1, tpr_m1, color='blue', lw=2, label=f'M1 ROC curve (area = {roc_auc_m1:.2f})')
plt.plot(fpr_m2, tpr_m2, color='red', lw=2, label=f'M2 ROC curve (area = {roc_auc_m2:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for M1 and M2')
plt.legend(loc="lower right")
plt.show()

# Determine which model performs better
better_model = 'M1' if roc_auc_m1 > roc_auc_m2 else 'M2'
print(f'M1 AUC: {roc_auc_m1:.2f}')
print(f'M2 AUC: {roc_auc_m2:.2f}')
print(f'The better model is {better_model}')