import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    true_positives = np.count_nonzero((prediction == ground_truth) & prediction)
    selected_elements = np.count_nonzero(prediction)
    relevant_elements = np.count_nonzero(ground_truth)
    true_elements = np.count_nonzero(prediction == ground_truth)
    
    precision = true_positives/selected_elements
    recall = true_positives/relevant_elements
    accuracy = true_elements/prediction.size
    f1 = 2*precision*recall/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.sum(prediction == ground_truth)/prediction.shape[0]
    
    return accuracy
