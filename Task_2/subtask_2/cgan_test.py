import torch
import numpy as np
from Task_2.subtask_1.routine import print_torch_version_info
from Task_2.subtask_1.data_preprocess import DataLoader
from Task_2.subtask_1.conditional_generative_adversarial_network import ConditionalGAN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Task_2.subtask_2.simple_perceptron import SimplePerceptron


def balance_dataset(data_loader, algorithm):
    (labels_data, counts_data) = np.unique(data_loader.y_labels, return_counts=True)
    max_count = np.argmax(counts_data)
    max_value = counts_data[max_count]
    latent_dim = algorithm.latent_dim

    for label, count in zip(labels_data, counts_data):
        # Generate noise
        if count == max_value:
            continue
        examples_count = max_value - count
        print("generating for label:", label, ",num examples:", examples_count)
        noise = np.random.random((examples_count, latent_dim))
        labels = np.full(examples_count, label)
        data_loader.y_labels = np.concatenate((data_loader.y_labels, labels), axis=0)
        generated_result = algorithm.generator.forward(noise, labels).detach().numpy()
        data_loader.X_data = np.concatenate((data_loader.X_data, generated_result), axis=0)


def train_classificator(algorithm_instance, *args):
    algorithm_instance.fit(*args)


def evaluate_classificator(algorithm_instance, x_data: np.ndarray, y_labels: np.ndarray, device=None):
    print(algorithm_instance, "algorithm results")

    if device is not None:
        y_pred = algorithm_instance.predict(x_data, device)
    else:
        y_pred = algorithm_instance.predict(x_data)

    print("Corresponding confusion matrix:")
    print(confusion_matrix(y_labels, y_pred), "\n")
    print("Corresponding classification report:")
    print(classification_report(y_labels, y_pred, zero_division=0))


def main():
    # Load trained cGAN model
    device = print_torch_version_info()

    # A dataset that will be balanced
    data_loader = DataLoader("../UNSW_NB15_testing-set.csv")
    # A dataset that will not be balanced
    unbalanced_data_loader = DataLoader("../UNSW_NB15_testing-set.csv")
    print("the data has been loaded with shape:", data_loader.X_data.shape, "y shape:", data_loader.y_labels.shape)

    # Create cGAN instance
    cgan = ConditionalGAN(input_shape=37, n_classes=10, latent_dim=50)
    # Load trained cGAN
    cgan.load_state_dict(torch.load("../model/model_cgan.pth", map_location=torch.device(device)))
    print("cGAN model has been loaded")

    # Balanced dataset using trained cGAN
    balance_dataset(data_loader, cgan)
    print("the data has been balanced, X shape:", data_loader.X_data.shape, "y shape:", data_loader.y_labels.shape)

    # Split datasets for clusterization algorithms
    x_train, x_test, y_train, y_test = train_test_split(
        data_loader.X_data, data_loader.y_labels, test_size=0.2, random_state=42)

    x_train_un, x_test_un, y_train_un, y_test_un = train_test_split(
        unbalanced_data_loader.X_data, unbalanced_data_loader.y_labels, test_size=0.2, random_state=42)

    print("{:-^70s}".format(" RANDOM FOREST CLASSIFIER "))
    # 1 classificator - Random Forest. We do not tune it
    random_forest_classifier = RandomForestClassifier()
    # train on unbalanced dataset
    train_classificator(random_forest_classifier, x_train_un, y_train_un)
    # test on unbalanced dataset
    print("{:-^70s}".format(" unbalanced dataset "))
    evaluate_classificator(random_forest_classifier, x_test_un, y_test_un)


    random_forest_classifier = RandomForestClassifier()
    # train on balanced dataset
    train_classificator(random_forest_classifier, x_train, y_train)
    # test on balanced dataset
    print("{:-^70s}".format(" balanced dataset "))
    evaluate_classificator(random_forest_classifier, x_test, y_test)

    print("{:-^70s}".format(" SIMPLE NEURAL NETWORK "))
    # 2 classificator - simple perceptron.
    simple_neural_network = SimplePerceptron(data_loader.X_data.shape[1], len(set(data_loader.y_labels))).to(device)
    simple_neural_network.compile(torch.optim.Adam)

    # train on unbalanced dataset
    train_classificator(simple_neural_network, 10, 32, x_train_un, y_train_un, device)
    # test on unbalanced dataset
    print("{:-^70s}".format(" unbalanced dataset "))
    evaluate_classificator(simple_neural_network, x_test_un, y_test_un, device)

    # train on balanced dataset
    simple_neural_network = SimplePerceptron(data_loader.X_data.shape[1], len(set(data_loader.y_labels))).to(device)
    simple_neural_network.compile(torch.optim.Adam)
    train_classificator(simple_neural_network, 10, 32, x_train, y_train, device)
    # test on balanced dataset
    print("{:-^70s}".format(" balanced dataset "))
    evaluate_classificator(simple_neural_network, x_test, y_test, device)


if __name__ == "__main__":
    main()
