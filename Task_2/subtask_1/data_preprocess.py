import pandas as pd
from os.path import exists
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler


class DataLoader:
    def __init__(self, network_intrusions_data_file):

        # Check file paths
        if not exists(network_intrusions_data_file):
            return

        self.scaled_dataframes = None
        self.X_data = None
        self.y_labels = None

        self.network_intrusions_data_file = network_intrusions_data_file

        # read data
        self.network_intrusions_data = pd.read_csv(network_intrusions_data_file)
        self.network_intrusions_column_length = len(self.network_intrusions_data)

        self._remove_incorrect_columns()

        # Set label values
        self.y_labels = self.network_intrusions_data["attack_cat"].to_numpy()
        self.network_intrusions_data = self.network_intrusions_data.drop(["attack_cat"], axis=1)

        self._encode_categorical_features()
        self._encode_y_labels()

        self._scale_data()

    def _encode_categorical_features(self):
        # select columns where type is not numerical
        categorical_features = self.network_intrusions_data.select_dtypes(include=['object'])

        cat_features_1d = categorical_features.to_numpy().ravel()

        # Encode them
        label_encoder = LabelEncoder()
        label_encoder.fit(cat_features_1d)

        for categorical_feature in categorical_features:
            encoded_column = label_encoder.transform(self.network_intrusions_data[categorical_feature])
            self.network_intrusions_data[categorical_feature] = encoded_column

    def _encode_y_labels(self):
        label_encoder = LabelEncoder()
        self.y_labels = label_encoder.fit_transform(self.y_labels)

    def _scale_data(self):
        scaler = RobustScaler()
        self.X_data = scaler.fit_transform(self.network_intrusions_data)

    def _remove_incorrect_columns(self, threshold=0.85):
        column_to_delete = ["id"]
        for feature in self.network_intrusions_data:
            feature_values_count = self.network_intrusions_data[feature].value_counts()
            for feature_freq in feature_values_count:
                if feature_freq / self.network_intrusions_column_length >= threshold:
                    column_to_delete.append(feature)

        self.network_intrusions_data = self.network_intrusions_data.drop(column_to_delete, axis=1)