import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from os.path import exists


class DataLoader:
    def __init__(self, identity_file, transaction_file, use_mask=True, preprocess_nans=True):

        # Check file paths
        if not exists(identity_file) or not transaction_file:
            return

        self.scaled_dataframes = None
        self.X_data = None
        self.y_labels = None
        self.use_mask = use_mask
        self.preprocess_nans = preprocess_nans
        # self.y_data = None

        self.identity_file = identity_file
        self.transaction_file = transaction_file

        # read data
        identity_data = pd.read_csv(identity_file)
        transaction_data = pd.read_csv(transaction_file)

        # Join our dataframes
        self.joined_dataframes_raw = pd.merge(left=transaction_data, right=identity_data, how='outer',
                                              left_on='TransactionID', right_on='TransactionID')

        # Drop transactionID column
        self.joined_dataframes_raw = self.joined_dataframes_raw.drop(["TransactionID"], axis=1)

        # Create a mask dataframe
        self.indexer = np.invert(self.joined_dataframes_raw.isnull()).astype(int)

        if preprocess_nans:
            self._encode_categorical_features()
        else:
            self._encode_categorical_features(nan_impute_value=np.nan)

        if self.preprocess_nans:
            self._scale_data()
            self._create_input_data(use_mask)


    def _encode_categorical_features(self, nan_impute_value=0):
        # select columns where type is not numerical
        categorical_features = self.joined_dataframes_raw.select_dtypes(include=['object'])

        cat_features_1d = categorical_features.to_numpy().ravel()

        # Encode them
        label_encoder = LabelEncoder()
        label_encoder.fit(cat_features_1d)
        le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        nan_number_encoded = le_name_mapping[np.nan]

        for categorical_feature in categorical_features:
            encoded_column = label_encoder.transform(self.joined_dataframes_raw[categorical_feature])
            self.joined_dataframes_raw[categorical_feature] = encoded_column
            self.joined_dataframes_raw[categorical_feature] = self.joined_dataframes_raw[categorical_feature].replace(nan_number_encoded, np.nan)

        if self.preprocess_nans:
            self.joined_dataframes_raw = self.joined_dataframes_raw.fillna(nan_impute_value)

        if not self.use_mask or not self.preprocess_nans:
            self.y_labels = self.joined_dataframes_raw["isFraud"]
            self.joined_dataframes_raw = self.joined_dataframes_raw.drop(["isFraud"], axis=1)

    def _scale_data(self):
        scaler = MinMaxScaler()
        self.scaled_dataframes = scaler.fit_transform(self.joined_dataframes_raw)

    def _create_input_data(self, use_mask=True):
        if use_mask:
            self.X_data = np.concatenate((self.scaled_dataframes, self.indexer), axis=1)
        else:
            self.X_data = self.scaled_dataframes




