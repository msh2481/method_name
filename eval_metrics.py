def calc_metrics_on_single(predicted, correct):
    """
    Calculate precision, recall, and F1 score for predicted and correct labels.

    Parameters:
    predicted (list of str): Predicted words.
    correct (list of str): Correct words.

    Returns:
    dict: A dictionary containing precision, recall, and F1 score.
    """
    predicted_set = set(predicted)
    correct_set = set(correct)

    true_positives = predicted_set & correct_set
    false_positives = predicted_set - correct_set
    false_negatives = correct_set - predicted_set

    precision = (
        len(true_positives) / (len(true_positives) + len(false_positives))
        if (len(true_positives) + len(false_positives)) > 0
        else 0
    )
    recall = (
        len(true_positives) / (len(true_positives) + len(false_negatives))
        if (len(true_positives) + len(false_negatives)) > 0
        else 0
    )
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {"precision": precision, "recall": recall, "f1_score": f1_score}


def calc_metrics(predicted_list, correct_list):
    """
    Apply the calculate_metrics function to lists of predicted and correct items.

    Parameters:
    predicted_list (list of list of str): List of predicted word lists.
    correct_list (list of list of str): List of correct word lists.

    Returns:
    list: A list of dictionaries containing precision, recall, and F1 score for each pair of predicted and correct items.
    """
    total_precision, total_recall, total_f1_score = 0, 0, 0
    n = len(predicted_list)

    for predicted, correct in zip(predicted_list, correct_list):
        metrics = calc_metrics_on_single(predicted, correct)
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1_score += metrics["f1_score"]

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_f1_score = total_f1_score / n

    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1_score": avg_f1_score,
    }


if __name__ == "__main__":
    predicted_list = [["labels", "count"], ["sum", "total"], ["get", "all"]]
    correct_list = [["count", "labels"], ["total", "sum"], ["get", "item"]]

    results = calc_metrics(predicted_list, correct_list)

    print(results)
