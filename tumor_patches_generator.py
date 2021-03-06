import os
import cv2
import csv
from tqdm import tqdm
from wsi_ops import WSIOps, PatchExtractor

base_path = '/atlas/home/zwpeng/paper_rebuild/camelyon/'

wsi_path  = 'alldatas/'
mask_path = 'train/tumor/annotation_images/'


def get_picture(path):
    path0 = os.path.join(base_path, path)
    picture = []
    for (_,_,filenames) in os.walk(path0):
        for filename in filenames:
            file_prefix = os.path.splitext(filename)[0]
            if os.path.exists(os.path.join(path0, file_prefix + ".tif")):
                picture.append(filename)
            else:
                print("wrong path or files")
    return picture

wsi_set = get_picture(wsi_path)
#mask_set = get_picture(mask_path)


def save_to_file(data, filename='bounding_boxes.txt'):
    if not os.path.exists(filename):
        os.mknod(filename)
    with open(filename) as f:
        f.write(data)
        f.close()
    print('saved successfully!')


csvfile = open('tumor_bounding_boxes_in_tumor_wsi.csv', 'w')
fieldnames = ['wsi', 'bounding_boxes', 'patch_index']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

wsioptions = WSIOps()
for i in tqdm(range(len(wsi_set))):
    wsi_mask, mask_image = wsioptions.read_wsi_mask(base_path + mask_path + os.path.splitext(wsi_set[i])[0] + "_Mask.tif")

    wsi_image, rgb_image, _, _, _ = wsioptions.read_wsi_tumor(base_path + wsi_path + wsi_set[i], base_path + mask_path + os.path.splitext(wsi_set[i])[0] + "_Mask.tif")

    bounding_boxes = wsioptions.find_roi_bbox_tumor_gt_mask(mask_image)
    print('%s bianjie' % os.path.splitext(wsi_set[i])[0], bounding_boxes)
#    writer.writerow({'wsi':os.path.splitext(wsi_set[i])[0], 'bounding_boxes':bounding_boxes})
    level_used = wsi_mask.level_count - 1
    patchex = PatchExtractor()
    patch_index = patchex.extract_positive_patches_from_tumor_region(wsi_image, mask_image, level_used, bounding_boxes, patch_save_dir='tumor_patches/', patch_prefix=os.path.splitext(wsi_set[i])[0] + '_', patch_index=0)
    print(patch_index)
    writer.writerow({'wsi':os.path.splitext(wsi_set[i])[0], 'bounding_boxes':bounding_boxes, 'patch_index':patch_index})

csvfile.close()

