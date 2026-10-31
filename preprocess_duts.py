import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_item(args):
    img_in, mask_in, img_out, mask_out = args
    if not os.path.exists(img_out):
        img = cv2.imread(img_in)
        if img is not None:
            resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_out, resized_img)
            
    if not os.path.exists(mask_out):
        mask = cv2.imread(mask_in, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            resized_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            # Binary mask: object is >= 128
            bin_mask = (resized_mask >= 128).astype(np.uint8) * 255
            cv2.imwrite(mask_out, bin_mask)

def preprocess_split(split_name):
    base_in = f'data/{split_name}'
    img_dir_in = os.path.join(base_in, f'{split_name}-Image')
    mask_dir_in = os.path.join(base_in, f'{split_name}-Mask')
    
    base_out = f'data/{split_name}-256'
    img_dir_out = os.path.join(base_out, 'images')
    mask_dir_out = os.path.join(base_out, 'masks')
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(mask_dir_out, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(img_dir_in) if not f.startswith('.')])
    tasks = []
    for f in img_files:
        name = os.path.splitext(f)[0]
        img_in = os.path.join(img_dir_in, f)
        mask_in = os.path.join(mask_dir_in, f'{name}.png')
        img_out = os.path.join(img_dir_out, f'{name}.png')
        mask_out = os.path.join(mask_dir_out, f'{name}.png')
        tasks.append((img_in, mask_in, img_out, mask_out))
        
    print(f'Preprocessing {split_name}: {len(tasks)} items...')
    with ProcessPoolExecutor() as executor:
        list(executor.map(process_item, tasks, chunksize=100))
    print(f'Finished {split_name}!')

if __name__ == '__main__':
    preprocess_split('DUTS-TR')
    preprocess_split('DUTS-TE')
