from src.mlModelsText.CustomNN import evaluate_model_customnn,predict_model_customnn
from src.mlModelsText.MultinomialNB import evaluate_model_nb,predict_model_nb
from src.mlModelsText.SVM import evaluate_model_svm,predict_model_svm
from src.preprocessing.text_treatment import csv_pipeline
from src.preprocessing.tfidfmatrix import createTfIdfMatrix_for_evaluation,processed_and_concatenated_X_Y
from src.utils.evaluation import evaluation_with_confusion_matrix
from scipy import sparse
import argparse
import pandas as pd
import sys


def main_model_evaluation(X_test_path,Y_test_path,path_root_output,model):
    path_of_the_output_tfidf_matrix=path_root_output+"TfIdfMatrix_evaluation.npz"
    path_of_the_processed_and_concatenated_X_Y=path_root_output+"processed_and_concatenated_X_Y_evaluation.csv"
    path_of_the_processed_X_Y=path_root_output+"processed_X_Y_evaluation.csv"
    evaluation=True

    dataframe_processed=csv_pipeline(X_test_path,evaluation,Y_test_path)
    dataframe_processed.to_csv(path_of_the_processed_X_Y, index=False)
    processed_and_concatenated_X_Y(path_of_the_processed_X_Y,path_of_the_processed_and_concatenated_X_Y)
    createTfIdfMatrix_for_evaluation(path_of_the_processed_and_concatenated_X_Y,path_of_the_output_tfidf_matrix)

    tfidfMatrix = sparse.load_npz(path_of_the_output_tfidf_matrix)
    dataframe_processed_and_concatenated = pd.read_csv(path_of_the_processed_and_concatenated_X_Y, index_col=0)
    if model=='multinomialnb':
        evaluate_model_nb(tfidfMatrix,dataframe_processed_and_concatenated["ProductTypeCode"])
        evaluation_with_confusion_matrix('./output/prediction_MultinomialNB.csv')
    if model=='customnn':
        evaluate_model_customnn(tfidfMatrix,dataframe_processed_and_concatenated["ProductTypeCode"])
        evaluation_with_confusion_matrix('./output/prediction_customNN.csv')
    if model=='svm':
        evaluate_model_svm(tfidfMatrix,dataframe_processed_and_concatenated["ProductTypeCode"])
        evaluation_with_confusion_matrix('./output/prediction_SVM.csv')
    else:
        "The model in argument hasn't been implemented yet !"


def convert_args_to_dict(args: argparse.Namespace) -> dict:
    """
    Convert argparse arguments into a dictionnary.

    From an argparse Namespace, create a dictionnary with only inputed CLI
    arguments. Allows to use argparse with default values in functions.

    parameters
    ----------
    args : argparse.Namespace
        Arguments to parse

    returns
    -------
    args_dict : dict
        Dictionnary with only inputed arguments
    """
    args_dict = {
        argument[0]: argument[1]
        for argument
        in args._get_kwargs()
        if argument[1] is not None}

    return args_dict


def parse_evaluate_args(
        args_to_parse) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--csv-X-test',
                        dest='X_test_path',
                        required=True)
    parser.add_argument('--csv-Y-test',
                        dest='Y_test_path',
                        required=True)
    parser.add_argument('--path-root',
                        dest='path_root_output',
                        required=True)
    parser.add_argument('--model',
                        dest='model',
                        required=True)
    args = parser.parse_args(args_to_parse)

    return args




if __name__ == '__main__':
    args = parse_evaluate_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    main_model_evaluation(**args_dict)