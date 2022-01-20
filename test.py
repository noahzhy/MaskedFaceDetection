import os
import cv2
import time
import argparse

import torch
import model.detector
import utils.utils

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/MFD.data',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='weights\MFD-70-epoch-0.550508ap-model.pth',
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str,
                        default=r'test/test00.jpg',
                        help='The path of test image')

    color_list = {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (0, 165, 255)
    }

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "Model weight path error!"
    assert os.path.exists(opt.img), "Test image path error!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(
        cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    # sets the module in eval node
    model.eval()

    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    # for grayscale
    # res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    # img = np.expand_dims(res_img, -1)
    # img = res_img.reshape(1, cfg["height"], cfg["width"], 1)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    start = time.perf_counter()
    preds = model(img)
    end = time.perf_counter()
    time = (end - start) * 1000.
    print("forward time: %fms" % time)

    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(
        output, conf_thres=0.2, iou_thres=0.0)
    print(output_boxes)

    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    for box in output_boxes[0]:
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]
        color = color_list[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, color, 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, color, 2)

    cv2.imwrite("test/{}_result.jpg".format(os.path.basename(opt.img).split('.')[0]), ori_img)
