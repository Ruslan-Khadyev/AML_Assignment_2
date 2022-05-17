import sys

import routine
from Task_1.data_preprocess import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def test_algorithm(x, y, algorithm):
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train algorithm
    algorithm.fit(x_train, y_train)

    # print results
    print(algorithm, "algorithm results")
    y_pred = algorithm.predict(x_test)
    print("Corresponding confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Corresponding classification report:")
    print(classification_report(y_test, y_pred))


def main(args):
    # Load data
    data_loader = DataLoader(args[1], args[1], use_mask=False)
    print("data has been loaded with X shape:", data_loader.X_data.shape, "y shape:", data_loader.y_labels.shape)

    # PCA
    routine.plot_explained_variance(data_loader.X_data, 50)
    pca = PCA(43)
    X_pca = pca.fit_transform(data_loader.X_data)
    print("PCA has been done")

    # LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(data_loader.X_data, data_loader.y_labels)
    print("LDA has been done")

    # Train algorithm
    dtc = DecisionTreeClassifier()

    test_algorithm(X_pca, data_loader.y_labels, dtc)
    test_algorithm(X_lda, data_loader.y_labels, dtc)


if __name__ == "__main__":
    main(sys.argv)

