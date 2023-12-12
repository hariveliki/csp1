import unittest
from utils import *


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # cls.df_cer = pd.read_csv("data/CER/data/xT_residential_25728.csv", low_memory=False)
        return super().setUpClass()


    def setUp(self) -> None:
        self.df_oocs = pd.read_csv("data/oocs/oocs_timestamp_ohe_one_year_pivoted.csv")
        self.df_oocs_labels = pd.read_csv("labels/oocs/boiler.csv")
        self.df_oozg = pd.read_csv("data/oozg/oozg_timestamp_ohe_one_year_pivoted.csv")
        self.path_x_train_cer = '/Users/haris.alic/Dev/github/r9119/csp1-koala/Team Koala/haris/appliance-detection-benchmark/data/CER/data/xT_residential_25728.csv'
        self.path_y_train_cer = '/Users/haris.alic/Dev/github/r9119/csp1-koala/Team Koala/haris/appliance-detection-benchmark/data/CER/labels/waterheater_case.csv'
        return super().setUp()


    def tearDown(self) -> None:
        del self.df_oocs
        del self.df_oozg
        return super().tearDown()


    def test_create_X_out_of_df(self):
        actual = create_X_out_of_df(self.df_oocs)
        self.assertIsInstance(actual, np.ndarray)


    def test_stack(self):
        X_1 = create_X_out_of_df(self.df_oocs)
        X_2 = create_X_out_of_df(self.df_oozg)
        actual = stack(X_1, X_2)
        self.assertIsInstance(actual, np.ndarray)
        self.assertGreater(actual.shape[0], X_1.shape[0])
        self.assertGreater(actual.shape[0], X_2.shape[0])


    def test_creat_X_y_out_of_df(self):
        Xy = load_transpose_CER(self.path_x_train_cer, self.path_y_train_cer)
        X, y = create_X_y_out_of_df(Xy)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(X.shape[0] == y.shape[0])
        self.assertTrue(y.shape[1] == 1)


    def test_creat_y_out_of_df(self):
        y = create_y_out_of_df(self.df_oocs_labels)
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(y.shape[1] == 1)


if __name__ == '__main__':
    unittest.main()