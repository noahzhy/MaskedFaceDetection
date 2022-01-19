import glob
import os
import pickle
import random
import xml.etree.ElementTree as ET
from os import getcwd, listdir
from os.path import join
from pathlib import Path
from shutil import copyfile, move
from PIL import Image


dirs = [
    r'D:\face_dataset\wider_face_add_lm_10_10/train',
    r'D:\face_dataset\wider_face_add_lm_10_10/val',
    # r'C:/Users/go/Desktop/MAFA/FMLD_annotations/train',
    # r'C:/Users/go/Desktop/MAFA/FMLD_annotations/test'
]
classes = list([i.replace('\n', '') for i in open('data/MFD.names', 'r')])
# print(classes)


def gen_train_val_list(path):
    train_val_list = ['train', 'val']
    for i in train_val_list:
        list_tv = glob.glob(os.path.join(path, i, '*.jpg'))
        with open(os.path.join(path, "{}.txt".format(i)), 'w') as output:
            for row in list_tv:
                output.write(str(row) + '\n')


def divide_dataset(root_path, train_target_dir, val_target_dir):
    total = glob.glob(os.path.join(root_path, '*.jpg'))
    random.shuffle(total)
    print("Total image number: {}".format(len(total)))
    train_set = random.sample(total, int(len(total) * 0.8))
    val_set = list(set(total) - set(train_set))

    for idx, i in enumerate(train_set):
        base_name = os.path.basename(i)
        base_name_no_exif = base_name.split('.')[0]
        dir_name = os.path.dirname(i)
        print(i)
        # if idx > 100: break
        copyfile(i, os.path.join(train_target_dir, base_name))
        move(os.path.join(dir_name, '{}.txt'.format(base_name_no_exif)), os.path.join(train_target_dir, '{}.txt'.format(base_name_no_exif)))

    for idx, i in enumerate(val_set):
        base_name = os.path.basename(i)
        base_name_no_exif = base_name.split('.')[0]
        dir_name = os.path.dirname(i)
        print(i)
        # if idx > 100: break
        copyfile(i, os.path.join(val_target_dir, base_name))
        move(os.path.join(dir_name, '{}.txt'.format(base_name_no_exif)), os.path.join(val_target_dir, '{}.txt'.format(base_name_no_exif)))


def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + '/' + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult)==1:
        #     continue
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        if b == 0 or w==0:
            im = Image.open(image_path)
            w, h = im.size
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    for dir_path in dirs:
        full_dir_path = dir_path
        output_path = full_dir_path +'/'

        image_paths = getImagesInDir(full_dir_path)
        print(len(image_paths))
        list_file = open(full_dir_path + '.txt', 'w')

        for image_path in image_paths:
            list_file.write(image_path + '\n')
            print(image_path)
            convert_annotation(dir_path, output_path, image_path)

        list_file.close()

    # original_path = r"C:\dataset\license_plate\license_plate_detection\JPEGImages"
    # train_target_dir = r"C:\dataset\license_plate\license_plate_detection\train"
    # val_target_dir = r"C:\dataset\license_plate\license_plate_detection\val"

    # Path(train_target_dir).mkdir(exist_ok=True, parents=True)
    # Path(val_target_dir).mkdir(exist_ok=True, parents=True)

    # divide_dataset(original_path, train_target_dir, val_target_dir)

    # gen_train_val_list("C:\dataset\license_plate\license_plate_detection")
    # print("Finished processing: " + dir_path)
