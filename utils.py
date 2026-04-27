
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def otsu_mask(image, size):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = th.astype(np.int32)
    th = th/255.0
    th = th > 0.5
    th = th.astype(np.int32)
    return th

def calculate_metrics(y_true, y_pred, num_classes=3):
    # ... (existing tensor conversion and flattening code) ...

    jaccard_scores, f1_scores, recall_scores, precision_scores = [], [], [], []
    epsilon = 1e-15
    
    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        
        # If the class is not in the ground truth, skip it for this image
        if true_c.sum() == 0:
            continue 
            
        tp = (true_c & pred_c).sum().float()
        fp = (~true_c & pred_c).sum().float()
        fn = (true_c & ~pred_c).sum().float()
            
        jaccard = tp / (tp + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
        jaccard_scores.append(jaccard)
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        
    # Average only over the classes that actually existed in this sample
    return [
        torch.stack(jaccard_scores).mean().item() if jaccard_scores else 0.0,
        torch.stack(f1_scores).mean().item() if f1_scores else 0.0,
        torch.stack(recall_scores).mean().item() if recall_scores else 0.0,
        torch.stack(precision_scores).mean().item() if precision_scores else 0.0
    ]