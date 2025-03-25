import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

spam_df = pd.read_csv("spambase_data.csv")
y_df = spam_df[['Class']].copy()
X_df = spam_df.copy()

X = X_df.to_numpy()
y = y_df.to_numpy()

proportion_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=proportion_test)

gnb = GaussianNB()

gnb.fit(X_train, y_train.ravel())

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión - Gaussian Naive Bayes')
plt.show()

accuracy = np.mean(y_pred == y_test)
sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print('Precisión: {:.4f}'.format(accuracy))
print('Sensibilidad (TPR): {:.4f}'.format(sensitivity))
print('Especificidad (TNR): {:.4f}'.format(specificity))