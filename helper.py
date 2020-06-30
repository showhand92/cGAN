import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class DisVectorData(Dataset):
    def __init__(self, file_path):
        self._db = self._get_db(file_path)
        print("data set shape: {}".format(self._db["vectors"].shape))
        print("max_v: {}, mav_a: {}".format(self._db["max_v"], self._db["max_a"]))

    def __getitem__(self, item):
        return self._db["vectors"][item, :], self._db["areas"][item]

    def __len__(self):
        return self._db["areas"].shape[-1]

    def _get_db(self, file_path):
        data = pd.read_excel(file_path, header=None)
        data = np.array(data)
        max_v = np.max(data[:, :-1])
        vectors = data[:, :-1]/max_v
        max_a = np.max(data[:, -1])*1.25
        areas = data[:, -1]/max_a
        db = {
            "max_v": max_v,
            "vectors": vectors,
            "max_a": max_a,
            "areas": areas,
            }
        return db

    def get_max_value(self):
        return self._db["max_v"], self._db["max_a"]
