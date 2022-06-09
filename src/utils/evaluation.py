"""
Example of custom metric script.
The custom metric script must contain the definition of custom_metric_function and a main function
that reads the two csv files with pandas and evaluate the custom metric.
"""

# TODO: add here the import necessary to your metric function
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sqlalchemy import true

def weighted_F1_score(dataframe_1, dataframe_2):
    y_dataframe_1 = np.array(dataframe_1["prdtypecode"])
    y_dataframe_2 = np.array(dataframe_2["prdtypecode"])

    score = f1_score(y_dataframe_1, y_dataframe_2, average="weighted")

    return score

def evaluation_with_confusion_matrix(path_of_the_output_dataframe):
    output_df=pd.read_csv(path_of_the_output_dataframe, names=['IntegerID','CodePredictions','RealProductTypeCodes'],skiprows=[0])
    inter_class_predictions={}
    inter_class_real={}
    prediction_vector=[]
    real_vector=[]
    labels=sorted(output_df['RealProductTypeCodes'].unique().tolist())
    labels_left=[]
    for k in range(len(labels)):
        inter_class_predictions[labels[k]]=0
        inter_class_real[labels[k]]=0
        if k%2==0:
            labels_left.append(labels[k])
    print(inter_class_predictions)
    j=0
    for k in range(len(output_df)):
        prediction_vector.append(output_df["CodePredictions"][k])
        real_vector.append(output_df["RealProductTypeCodes"][k])
            
        #inter_class_good_predictions[output_df["CodePredictions"][k]]+=1
        inter_class_real[output_df["RealProductTypeCodes"][k]]+=1
    print(inter_class_predictions)
    print(j)
    prediction_vector=np.array(prediction_vector)
    real_vector=np.array(real_vector)
    print('F1 score is ',f1_score(real_vector,prediction_vector, average="weighted"))             #F1 = 2 * (precision * recall) / (precision + recall)
    print('Accuracy is ',accuracy_score(real_vector,prediction_vector))       #In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print('Recall is ',recall_score(real_vector,prediction_vector,average="weighted"))         
    print('Precision is ',precision_score(real_vector,prediction_vector,average="weighted"))   

    array=confusion_matrix(real_vector,prediction_vector)
    df_res=pd.DataFrame(array)
    ax=plt.subplot()
    sn.heatmap(df_res, cmap=sn.cm.rocket_r, fmt='g', annot=True,vmin=0)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels'); 
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels_left)
    plt.show()


if __name__ == '__main__':
    evaluation_with_confusion_matrix('./output/prediction_MultinomialNB.csv')


"""
if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_1 = 'y_test.csv'
    CSV_FILE_2 = 'Y_test_benchmark_text.csv'
    df_1 = pd.read_csv(CSV_FILE_1, index_col=0, sep=',')
    df_2 = pd.read_csv(CSV_FILE_2, index_col=0, sep=',')
    print(weighted_F1_score(df_1, df_2))
"""