import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from model import build_doubleunet
from utils import create_dir, seeding, calculate_metrics
from train import load_data

def process_mask(y_pred_classes):
    # --- MODIFIED FOR MULTI-CLASS ---
    # Input is now discrete classes (0, 1, 2), not probabilities
    y_pred = y_pred_classes[0].cpu().numpy()
    
    # Scale classes to grayscale pixel values so we can see them in saved images
    # 0 = Background (Black), 1 * 127 = 127 Class 1 (Gray), 2 * 127 = 254 Class 2 (White)
    y_pred = (y_pred * 127).astype(np.uint8)
    
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

# FIX: Updated to expect 4 metrics instead of 6
def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0]/num_samples
    f1 = metrics_score[1]/num_samples
    recall = metrics_score[2]/num_samples
    precision = metrics_score[3]/num_samples

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f}")

def evaluate(model, save_path, test_x, test_y, size):
    # FIX: Arrays now hold 4 zeros to match utils.py
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0]
    metrics_score_2 = [0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x)

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image/255.0
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        # FIX: Ported the mapping logic from train.py so mask becomes 0, 1, or 2
        mask_raw = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)
        
        final_mask = np.zeros(mask_raw.shape, dtype=np.int64)
        filename = os.path.basename(y).lower()
        
        tumor_pixels = (mask_raw > 127)
        if "benign" in filename:
            final_mask[tumor_pixels] = 1      # Class 1: Benign
        elif "malignant" in filename:
            final_mask[tumor_pixels] = 2      # Class 2: Malignant

        # Scale for saving the visual representation properly
        save_mask = (final_mask * 127).astype(np.uint8) 
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        
        # Convert to tensor properly for multi-class (Integer indices, Shape: [1, H, W])
        mask = torch.from_numpy(final_mask).long().unsqueeze(0) 
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            y_pred1, y_pred2 = model(image)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            # --- MODIFIED FOR MULTI-CLASS ---
            y_pred1 = torch.softmax(y_pred1, dim=1)
            y_pred2 = torch.softmax(y_pred2, dim=1)

            y_pred1_classes = torch.argmax(y_pred1, dim=1)
            y_pred2_classes = torch.argmax(y_pred2, dim=1)

            """ Evaluation metrics """
            mask_flat = mask.squeeze(0) 

            score_1 = calculate_metrics(mask_flat, y_pred1_classes)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))

            score_2 = calculate_metrics(mask_flat, y_pred2_classes)
            metrics_score_2 = list(map(add, metrics_score_2, score_2))

            """ Predicted Mask """
            y_pred1_img = process_mask(y_pred1_classes)
            y_pred2_img = process_mask(y_pred2_classes)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, y_pred1_img, line, y_pred2_img], axis=1)
        cv2.imwrite(f"{save_path}/joint/{name}", cat_images)
        cv2.imwrite(f"{save_path}/mask1/{name}", y_pred1_img)
        cv2.imwrite(f"{save_path}/mask2/{name}", y_pred2_img)

    print("--- Output 1 Scores ---")
    print_score(metrics_score_1, len(test_x))
    
    print("--- Output 2 Scores (Final Output) ---")
    print_score(metrics_score_2, len(test_x))

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_doubleunet()
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "dataset_seg"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    save_path = f"results"
    for item in ["mask1", "mask2", "joint"]:
        create_dir(f"{save_path}/{item}")

    size = (256, 256)
    evaluate(model, save_path, test_x, test_y, size)