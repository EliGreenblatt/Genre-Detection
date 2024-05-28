from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from classifier_interface import ClassifierInterface
import matplotlib.pyplot as plt
import seaborn as sns

class SVMClassifier(ClassifierInterface):
    def __init__(self, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy, confusion_matrix(y_test, predictions)

    def __str__(self):
        return f"SVM Classifier with C={self.model.C} and kernel='{self.model.kernel}'"

if __name__ == '__main__':
    kernel = 'rbf'
    C = 1.0

    file_paths = [
                  '../../features/mfcc/features_29032024_1930.csv']
    n_runs = 20
    label_names = {0: 'Classical', 1: 'Jazz', 2: 'Metal', 3: 'Pop', 4: 'Rock',5: 'Electronic'
                   , 6: 'Disco', 7: 'Blues', 8: 'Reggae', 9: 'Hiphop', 10: 'Country'}

    results = {}

    for file_path in file_paths:
        print(f"Evaluating file: {file_path}")
        # Ensure this function is defined or imported correctly
        X, y = SVMClassifier.load_data(file_path)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        accuracies = []
        for seed in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=seed)
            svm_classifier = SVMClassifier(C=C, kernel=kernel)
            svm_classifier.train(X_train, y_train)
            accuracy, confusion = svm_classifier.evaluate(X_test, y_test)
            accuracies.append(accuracy)
 

        average_accuracy = sum(accuracies) / n_runs
        results[file_path] = average_accuracy
             # Display confusion matrix
        plt.figure()
        sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g',
                                xticklabels=[label_names[i] for i in range(len(label_names))],
                                yticklabels=[label_names[i] for i in range(len(label_names))])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    print()
    print("Overall Results:")
    for file_path in file_paths:
        print(f"File: {file_path}")
        print(f"The Accuracy of predicting the song genre with the SVM algorithm and the given parameters is: {results[file_path]:.2f}%")