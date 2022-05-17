import pandas as pd
import numpy as np
from Task_1.data_preprocess import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from Task_1.subtask_1.Undercomplete_autoencoder import UnderCompleteAutoencoder
import torch


def main():
    # Load the data without prepared nan values
    print("=========== Statistical imputing approach ===========")
    data_loader = DataLoader("../data_identity.csv", "../data_transaction.csv", preprocess_nans=False)
    print(data_loader.joined_dataframes_raw.head(5))
    print("Data has been loaded")

    # Now we will impute values with statistics or median strategy (the best of the tested)
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    x_imputed_stat = simple_imputer.fit_transform(data_loader.joined_dataframes_raw)
    print("Data has been imputed with 'median' strategy")

    # Scale our imputed data
    stat_scal = MinMaxScaler()
    x_scaled_stat = stat_scal.fit_transform(x_imputed_stat)
    print("Data has been scaled")

    # Split to test and train with 80% / 20% ratio
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_stat, data_loader.y_labels, test_size=0.2,
                                                        random_state=42)
    print("Data has been splitted")

    # as a test algorithm Decision Tree Classifier will be applied
    dtc = DecisionTreeClassifier()
    print("Classifier has been created")
    dtc.fit(x_train, y_train)
    # Finally, let's print the results
    print(dtc, "algorithm classification results")
    y_pred = dtc.predict(x_test)
    print("Corresponding confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Corresponding classification report:")
    print(classification_report(y_test, y_pred))
    print()

    print("=========== Autoencoder imputing approach ===========")
    # Let's load Undercoplete autoencoders
    au = UnderCompleteAutoencoder(866, 10)
    au.load_state_dict(torch.load("../subtask_1/autoencoders/UnderComplete.pth", map_location=torch.device('cpu')))
    print("Encoder has been loaded")
    data_loader_encoder = DataLoader("../data_identity.csv", "../data_transaction.csv")
    print("Data has been reloaded")
    # fill missed values with autoencoders
    filled = au.forward(torch.from_numpy(data_loader_encoder.X_data).float()).detach().numpy()
    print("Data has been imputed with autoencoder")

    # split filled with autoencoder data
    x_train, x_test, y_train, y_test = train_test_split(filled, data_loader.y_labels, test_size=0.2, random_state=42)
    print("Data has been splitted")

    dtc = DecisionTreeClassifier()
    print("Classifier has been recreated")
    dtc.fit(x_train, y_train)

    # print result information
    print(dtc, "algorithm results")
    y_pred = dtc.predict(x_test)
    print("Corresponding confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Corresponding classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()