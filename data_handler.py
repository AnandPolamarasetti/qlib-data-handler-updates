import unittest
import pytest
import sys
from qlib.tests import TestAutoData
from qlib.data.dataset import TSDatasetH, TSDataSampler
import numpy as np
import pandas as pd
import time
from qlib.data.dataset.handler import DataHandlerLP


class TestDataset(TestAutoData):
    @pytest.mark.slow
    def test_TSDataset(self):
        tsdh = TSDatasetH(
            handler={
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2017-01-01",
                    "end_time": "2020-08-01",
                    "fit_start_time": "2017-01-01",
                    "fit_end_time": "2017-12-31",
                    "instruments": "csi300",
                    "infer_processors": [
                        {"class": "FilterCol", "kwargs": {"col_list": ["RESI5", "WVMA5", "RSQR5"]}},
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [
                        "DropnaLabel",
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                },
            },
            segments={
                "train": ("2017-01-01", "2017-12-31"),
                "valid": ("2018-01-01", "2018-12-31"),
                "test": ("2019-01-01", "2020-08-01"),
            },
        )

        # Use numpy.random.Generator
        rng = np.random.default_rng()

        tsds_train = tsdh.prepare("train", data_key=DataHandlerLP.DK_L)
        tsds = tsdh.prepare("valid", data_key=DataHandlerLP.DK_L)

        t = time.time()
        for idx in rng.integers(0, len(tsds_train), size=2000):
            _ = tsds_train[idx]
        print(f"2000 samples took {time.time() - t:.2f} seconds")

        t = time.time()
        for _ in range(20):
            data = tsds_train[rng.integers(0, len(tsds_train), size=2000)]
        print(data.shape)
        print(f"2000 samples (batch index) * 20 times took {time.time() - t:.2f} seconds")

        # Check the data consistency using different sampling methods

        tsds[len(tsds) - 1]

        data_from_ds = tsds["2017-12-31", "SZ300315"]

        data_from_df = (
            tsdh.handler.fetch(data_key=DataHandlerLP.DK_L)
            .loc(axis=0)["2017-01-01":"2017-12-31", "SZ300315"]
            .iloc[-30:]
            .values
        )

        equal = np.isclose(data_from_df, data_from_ds, equal_nan=True)
        self.assertTrue(equal[~np.isnan(data_from_df)].all())


class TestTSDataSampler(unittest.TestCase):
    def test_TSDataSampler(self):
        datetime_list = ["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30", "2000-05-31"]
        instruments = ["000001", "000002", "000003", "000004", "000005"]
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(datetime_list), instruments], names=["datetime", "instrument"]
        )
        data = np.random.randn(len(datetime_list) * len(instruments))
        test_df = pd.DataFrame(data=data, index=index, columns=["factor"])
        dataset = TSDataSampler(test_df, datetime_list[0], datetime_list[-1], step_len=2)

        print("\n--------------dataset[0]--------------")
        print(dataset[0])
        print("--------------dataset[1]--------------")
        print(dataset[1])

        self.assertEqual(len(dataset[0]), 2)
        self.assertTrue(np.isnan(dataset[0][0]))
        self.assertEqual(dataset[0][1], dataset[1][0])
        self.assertEqual(dataset[1][1], dataset[2][0])
        self.assertEqual(dataset[2][1], dataset[3][0])

    def test_TSDataSampler2(self):
        datetime_list = ["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30", "2000-05-31"]
        instruments = ["000001", "000002", "000003", "000004", "000005"]
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(datetime_list), instruments], names=["datetime", "instrument"]
        )
        data = np.random.randn(len(datetime_list) * len(instruments))
        test_df = pd.DataFrame(data=data, index=index, columns=["factor"])
        dataset = TSDataSampler(test_df, datetime_list[2], datetime_list[-1], step_len=3)

        print("\n--------------dataset[0]--------------")
        print(dataset[0])
        print("--------------dataset[1]--------------")
        print(dataset[1])

        for i in range(3):
            self.assertFalse(np.isnan(dataset[0][i]))
            self.assertFalse(np.isnan(dataset[1][i]))
        self.assertEqual(dataset[0][1], dataset[1][0])
        self.assertEqual(dataset[0][2], dataset[1][1])


if __name__ == "__main__":
    unittest.main(verbosity=10)
