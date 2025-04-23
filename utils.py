import torch


def calculate_tfpn(edge_predictions, edge_labels):
    edge_predictions = torch.round(torch.sigmoid(edge_predictions))
    TP = torch.sum(torch.logical_and(edge_predictions == 1, edge_labels == 1)).item()
    TN = torch.sum(torch.logical_and(edge_predictions == 0, edge_labels == 0)).item()
    FP = torch.sum(torch.logical_and(edge_predictions == 1, edge_labels == 0)).item()
    FN = torch.sum(torch.logical_and(edge_predictions == 0, edge_labels == 1)).item()
    return TP, TN, FP, FN


def calculate_metrics(TP, TN, FP, FN):
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = TP / (TP + 0.5 * (FP + FN))
    except ZeroDivisionError:
        f1 = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, precision, recall, f1


def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'
