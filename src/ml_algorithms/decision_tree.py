from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from classifier_interface import ClassifierInterface


class DTClassifier(ClassifierInterface):
    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def __str__(self):
        return f"Decision Tree Classifier with max_depth={self.model.max_depth}, criterion={self.model.criterion}, min_samples_split={self.model.min_samples_split}, min_samples_leaf={self.model.min_samples_leaf}"


def main(max_depth, criterion, min_samples_split, min_samples_leaf):
    file_paths = ['../../features/chroma/features_29032024_1938.csv',
                  '../../features/mfcc/features_29032024_1930.csv']
    n_runs = 20

    results = {}

    for file_path in file_paths:
        print(f"Evaluating file: {file_path}")
        X, y = DTClassifier.load_data(file_path)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        accuracies = []
        for seed in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=seed)
            dt = DTClassifier(max_depth=max_depth, criterion=criterion,
                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            dt.train(X_train, y_train)
            accuracy = dt.evaluate(X_test, y_test)
            accuracies.append(accuracy)
        average_accuracy = sum(accuracies) / n_runs
        results[file_path] = average_accuracy

    print("\nOverall Results:")
    for file_path in file_paths:
        print(f"File: {file_path}")
        print(f"Max Depth: {max_depth}, Criterion: {criterion}, Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}")
        print(f"Accuracy: {results[file_path]:.2f}")