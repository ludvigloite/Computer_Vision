import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    # Compute intersection
    x_min_point = np.maximum(prediction_box[0],gt_box[0])
    x_max_point = np.minimum(prediction_box[2],gt_box[2])
    y_min_point = np.maximum(prediction_box[1],gt_box[1])
    y_max_point = np.minimum(prediction_box[3],gt_box[3])

    x_len = x_max_point - x_min_point
    y_len = y_max_point - y_min_point

    intersection = x_len*y_len
    if x_len<0 or y_len<0:
        intersection = 0

    # Compute union
    #union = predicted - gt - intersection
    predicted_area = (prediction_box[2]-prediction_box[0])*(prediction_box[3]-prediction_box[1])
    gt_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])

    union = predicted_area + gt_area - intersection

    iou = intersection / union

    assert iou >= 0 and iou <= 1

    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp+num_fp==0:
        return 1
    return num_tp/(num_tp+num_fp)

    


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp+num_fp==0:
        return 0
    return num_tp/(num_tp+num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    
    nuPredBoxes = prediction_boxes.shape[0]
    nuGTBoxes = gt_boxes.shape[0]

    allMatches = np.zeros(shape=(nuPredBoxes*nuGTBoxes, 3)) #predBoxNR, GTBoxNR, iou

    # Find all possible matches with a IoU >= iou threshold
    k = 0
    for i in range(nuPredBoxes):
        for j in range(nuGTBoxes):
            iou = calculate_iou(prediction_boxes[i],gt_boxes[j])
            if iou >= iou_threshold:
                allMatches[k] = [i, j, iou]
                k+=1

    # Sort all matches on IoU in descending order
    sortedMatches = allMatches[allMatches[:,2].argsort()[::-1]]

    # Find all matches with the highest IoU threshold
    returnPredArray = []
    returnGtArray = []
    predIndicesTaken = []
    gtIndicesTaken = []

    for l in range(sortedMatches.shape[0]):
        match = allMatches[l]
        if match[0] not in predIndicesTaken and match[1] not in gtIndicesTaken and match[2]>=iou_threshold:
            returnPredArray.append(prediction_boxes[int(match[0])])
            returnGtArray.append(gt_boxes[int(match[1])])
            predIndicesTaken.append([match[0]])
            gtIndicesTaken.append([match[1]])
    #print(f"predArray= {returnPredArray}. GtArray= {returnGtArray}")
    return np.array(returnPredArray), np.array(returnGtArray)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    
    _, alignedGtBoxes = get_all_box_matches(prediction_boxes,gt_boxes,iou_threshold)
    
    nuTruePos = np.unique(alignedGtBoxes, axis = 0).shape[0]
    nuFalsePos = prediction_boxes.shape[0]-nuTruePos
    nuFalseNeg = gt_boxes.shape[0]-nuTruePos

    result = {}
    result["true_pos"] = nuTruePos
    result["false_pos"] = nuFalsePos
    result["false_neg"] = nuFalseNeg

    #print(result)

    return result
    


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(all_prediction_boxes)):
        result = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        tp += result["true_pos"]
        fp += result["false_pos"]
        fn += result["false_neg"]

    precision = calculate_precision(tp,fp,fn)
    recall = calculate_recall(tp,fp,fn)
    #print(precision,recall)
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    precisions = [] 
    recalls = []

    #go though the iterations of conf_thres and check precision and recalls at this steps
    for conf_threshold in confidence_thresholds:
        pred_boxes_inside_conf = []
        for i in range(len(confidence_scores)):
            boxes_inside_conf_bool = confidence_scores[i] >= conf_threshold
            pred_boxes_inside_conf.append(all_prediction_boxes[i][boxes_inside_conf_bool,:])

        prec_and_recalls = calculate_precision_recall_all_images(pred_boxes_inside_conf,all_gt_boxes,iou_threshold)

        precisions.append(prec_and_recalls[0])
        recalls.append(prec_and_recalls[1])

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0]) #var 0.8,1.0
    plt.ylim([0.8, 1.0]) #var 0.8,1.0
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    plot_precision_recall_curve(precisions,recalls)
    
    precision_vals = []
    for recall_level in recall_levels:
        precs = []
        for precicion, recall in zip(precisions,recalls):
            if recall >= recall_level:
                precs.append(precicion)
        if len(precs)>0:
            precision_vals.append(max(precs))
        else:
            precision_vals.append(0)

    average_precision = np.mean(precision_vals)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
