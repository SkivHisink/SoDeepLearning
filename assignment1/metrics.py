def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    temp_tp = 0
    temp_tn = 0
    temp_fn = 0
    temp_fp = 0
    for i in range(0, len(prediction)):
        temp_tp += prediction[i] == True and ground_truth[i] == True if 1 else 0
        temp_fp += prediction[i] == True and ground_truth[i] == False if 1 else 0
        temp_tn += prediction[i] == False and ground_truth[i] == False if 1 else 0
        temp_fp += prediction[i] == False and ground_truth[i] == True if 1 else 0
    tp = temp_tp / len(prediction)
    tn = temp_tn / len(prediction)
    fn = temp_fn / len(prediction)
    fp = temp_fp / len(prediction)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    tp = 0
    for i in range(len(prediction)):
            tp += prediction[i] == ground_truth[i] if 1 else 0
    return tp / len(prediction)
