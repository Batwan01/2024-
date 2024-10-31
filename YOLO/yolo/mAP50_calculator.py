import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
        
    return intersection / union

def parse_boxes(prediction_string):
    if pd.isna(prediction_string):
        return []
    boxes = []
    parts = prediction_string.strip().split()
    for i in range(0, len(parts), 6):
        if i + 6 <= len(parts):
            box = [float(x) for x in parts[i:i+6]]
            boxes.append(box)
    return boxes

def calculate_map50(gt_path, pred_path):
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    CLASS_NAMES = ['veh_go', 'veh_goLeft', 'veh_noSign', 'veh_stop',
                   'veh_stopLeft', 'veh_stopWarning', 'veh_warning',
                   'ped_go', 'ped_noSign', 'ped_stop',
                   'bus_go', 'bus_noSign', 'bus_stop', 'bus_warning']
    
    # 이미지 매칭 확인
    gt_images = set(gt_df['image_id'])
    pred_images = set(pred_df['image_id'])
    print(f"Ground truth images: {len(gt_images)}")
    print(f"Prediction images: {len(pred_images)}")
    print(f"Matched images: {len(gt_images.intersection(pred_images))}")
    
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    for _, row in gt_df.iterrows():
        boxes = parse_boxes(row['PredictionString'])
        all_ground_truths[row['image_id']] = boxes
        
    for _, row in pred_df.iterrows():
        boxes = parse_boxes(row['PredictionString'])
        boxes.sort(key=lambda x: x[1], reverse=True)
        all_predictions[row['image_id']] = boxes
    
    aps = []
    class_aps = {}
    
    for class_id in range(14):
        true_positives = []
        false_positives = []
        scores = []
        n_ground_truths = 0
        
        for gt_boxes in all_ground_truths.values():
            n_ground_truths += sum(1 for box in gt_boxes if box[0] == class_id)
        
        if n_ground_truths == 0:
            print(f"No ground truths for class {class_id}")
            continue
            
        for image_id in all_ground_truths.keys():
            gt_boxes = [box for box in all_ground_truths[image_id] if box[0] == class_id]
            pred_boxes = [box for box in all_predictions[image_id] if box[0] == class_id]
            
            gt_matched = [False] * len(gt_boxes)
            
            for pred_box in pred_boxes:
                scores.append(pred_box[1])
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                        
                    iou = calculate_iou(pred_box[2:], gt_box[2:])
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    true_positives.append(1)
                    false_positives.append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    true_positives.append(0)
                    false_positives.append(1)
        
        if not scores:
            print(f"No predictions for class {class_id}")
            continue
            
        scores = np.array(scores)
        true_positives = np.array(true_positives)
        false_positives = np.array(false_positives)
        
        sorted_indices = np.argsort(-scores)
        true_positives = true_positives[sorted_indices]
        false_positives = false_positives[sorted_indices]
        
        cumsum_tp = np.cumsum(true_positives)
        cumsum_fp = np.cumsum(false_positives)
        
        recalls = cumsum_tp / n_ground_truths
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp)
        
        ap = 0
        prev_recall = 0
        for i in range(len(precisions)):
            if i == 0 or recalls[i] != recalls[i-1]:
                ap += (recalls[i] - prev_recall) * precisions[i]
                prev_recall = recalls[i]
        
        aps.append(ap)
        class_aps[class_id] = ap
        print(f"Class {CLASS_NAMES[class_id]} AP: {ap:.4f}")
    
    mAP = np.mean(aps)
    print(f"\nmAP@50: {mAP:.4f}")
    
    return mAP, class_aps

if __name__ == "__main__":
    gt_path = "/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/csv/val_ground_truth.csv"
    pred_path = "/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/ensemble/output/nmw_ensemble.csv"
    
    mAP, _ = calculate_map50(gt_path, pred_path)
    print(pred_path[pred_path.rfind('/')+1:])