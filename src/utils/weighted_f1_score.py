"""
Example of custom metric script.
The custom metric script must contain the definition of custom_metric_function and a main function
that reads the two csv files with pandas and evaluate the custom metric.
"""

# TODO: add here the import necessary to your metric function
import numpy as np
from sklearn.metrics import f1_score

def weighted_F1_score(dataframe_1, dataframe_2):
    y_dataframe_1 = np.array(dataframe_1["prdtypecode"])
    y_dataframe_2 = np.array(dataframe_2["prdtypecode"])

    score = f1_score(y_dataframe_1, y_dataframe_2, average="weighted")

    return score


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_1 = 'y_test.csv'
    CSV_FILE_2 = 'Y_test_benchmark_text.csv'
    df_1 = pd.read_csv(CSV_FILE_1, index_col=0, sep=',')
    df_2 = pd.read_csv(CSV_FILE_2, index_col=0, sep=',')
    print(weighted_F1_score(df_1, df_2))
