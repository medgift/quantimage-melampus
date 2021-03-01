import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder

from melampus.melampus_base import Melampus

class MelampusPreprocessor(Melampus):
    """
    This is the preprocessor class for melampus platform. It process data in order to be standarized and normalized.
    It also provides the method for dimensionality reduction and the removal of highly correlated features.
    See the Melampus Class for information about input parameters.
    """

    def standarize_data(self):
        """
        Standarization of the data using scikit-learn ``StandardScaler``.
        """

        scaler = StandardScaler().fit(self.data)
        self.data = scaler.transform(self.data)

    def normalize_data(self):
        """
        Normalize data with L2 norm using ``normalize`` method from scikit-learn
        """

        self.data = normalize(self.data)

    def dimensionality_reduction(self, num_components: int):
        """
        Reduce the amount of features into a new feature space using Principal Component Analysis.

        :param num_components: number of dimentions in the new feature space
        :type num_components: int, required
        """
        pca = PCA(n_components=num_components)
        self.data = pca.fit_transform(self.data)

