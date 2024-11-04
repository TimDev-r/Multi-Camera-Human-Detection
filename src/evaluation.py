def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxA[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate(detections, ground_truths, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for gt in ground_truths:
        gt_bbox = gt['bbox']
        gt_class_id = gt['class_id']

        matched = False
        for det in detections:
            det_bbox = det['bbox']
            det_class_id = det['class_id']
            if det_class_id == gt_class_id and calculate_iou(gt_bbox, det_bbox) >= iou_threshold:
                tp += 1
                matched = True
                break

        if not matched:
            fn += 1

    fp = len(detections) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score
