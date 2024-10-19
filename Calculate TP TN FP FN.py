import numpy as np

true_class = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
prob_m1 = np.array([0.50, 0.69, 0.44, 0.55, 0.67, 0.47, 0.08, 0.15, 0.45, 0.35])
prob_m2 = np.array([0.61, 0.03, 0.68, 0.31, 0.45, 0.09, 0.38, 0.05, 0.01, 0.04])

def calculate_confusion_matrix(true_class, prob, threshold):
    true_class = np.array(true_class)
    prob = np.array(prob)
    
    tp = np.sum((prob >= threshold) & (true_class == 1))
    tn = np.sum((prob < threshold) & (true_class == 0))
    fp = np.sum((prob >= threshold) & (true_class == 0))
    fn = np.sum((prob < threshold) & (true_class == 1))
    
    
    return tp, tn, fp, fn

# Example usage
threshold = 0.01
tp, tn, fp, fn = calculate_confusion_matrix(true_class, prob_m2, threshold)
TPR = tp/(tp+fn)
FPR = fp/(fp+tn)
print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, TPR: {TPR}, FPR: {FPR}')