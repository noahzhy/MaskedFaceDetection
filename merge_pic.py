import os
import cv2
import time
from glob import glob

import torch
import numpy as np
from PIL import Image

import utils.utils
import model.detector


weights='weights/LPD-120-epoch-0.792801ap-model.pth'
cfg = utils.utils.load_datafile('data/LPD.data')
device = torch.device("cpu")
model = model.detector.Detector(2, 3, True).to(device)
model.load_state_dict(torch.load(weights, map_location=device))
# sets the module in eval node
model.eval()

crop_offset_h, crop_offset_w = 20, 20


def mini_size(img_path):
    MAX_SIZE = (256, 256) 
    im = Image.open(img_path).thumbnail(MAX_SIZE).save(img_path)


def get_file_name(file_path) -> str:
    filename = os.path.basename(file_path).split('.')[0]
    lp = filename.split('-')[0]
    return lp


def save_file(origin_name, save_dir, img=None):
    timestamp = int(round(time.time() * 1000))
    save_path = '{}_{}.jpg'.format(origin_name, timestamp)

    full_save_path = os.path.join(save_dir, save_path)
    cv2.imencode('.jpg', img)[1].tofile(full_save_path)


def adj_resize(image, fixed_height=None, fixed_width=None):
    if fixed_height:
        height_percent = (fixed_height / float(image.size[1]))
        width_size = int((float(image.size[0]) * float(height_percent)))
        image = image.resize((width_size, fixed_height), Image.ANTIALIAS)
    if fixed_width:
        width_percent = (fixed_width / float(image.size[0]))
        height_size = int((float(image.size[1]) * float(width_percent)))
        image = image.resize((fixed_width, height_size), Image.ANTIALIAS)
    return image


def merge_pic(image, background='background.jpg'):
    im = Image.open(image)
    im = adj_resize(im, fixed_height=360)
    w, h = im.size
    off_w, off_h = w//2, h//2
    bg = Image.open(background)
    bg.paste(im, (1173-off_w, 1075-off_h))
    return bg


def infer_img(img, origin_name, save_dir):
    ori_img = np.array(img)
    print("ori_img.shape", ori_img.shape)
    res_img = cv2.resize(ori_img, (352, 352), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, 352, 352, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    preds = model(img)

    # 特征图后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.85, iou_thres=0)

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / 352, w / 352

    for box in output_boxes[0]:
        box = box.tolist()
        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        print(x1, y1, x2, y2)
        cropped = ori_img[y1-crop_offset_h:y2+crop_offset_h, x1-crop_offset_w:x2+crop_offset_w]
        cropped = adj_resize(Image.fromarray(cropped), fixed_width=256)
        cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        save_file(origin_name, save_dir, cropped)
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey(0)
    

if __name__ == '__main__':
    # im_dir = r'C:\dataset\license_plate\license_plate_recognition\aihub_crop\test'
    im_dir = r'C:\dataset\license_plate\license_plate_recognition\aihub'
    save_dir = r'C:\dataset\license_plate\license_plate_recognition\aihub_crop\save'

    for file_path in glob(os.path.join(im_dir, '*/*.jpg')):
        f_name = get_file_name(file_path)
        new_img = merge_pic(file_path)
        infer_img(new_img, f_name, save_dir)
        print(f_name)
