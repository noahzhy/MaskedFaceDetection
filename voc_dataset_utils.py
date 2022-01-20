import glob
import os
import random
import xml.etree.ElementTree as ET
from glob import glob
from os import path, walk
from pathlib import Path
from shutil import copyfile
from xml.etree.ElementTree import ElementTree

from PIL import Image


def png2jpg(im_path):
    for i in glob(os.path.join(im_path, '*.png')):
        im = Image.open(i).convert('RGB')
        im.save(i.replace('.png', '.jpg'))


def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def write_xml(tree, out_path):
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def get_path_prex(rootdir):
    data_path = []
    prefixs = []
    for root, dirs, files in walk(rootdir, topdown=True):
        for name in files:
            pre, ending = path.splitext(name)
            if ending != ".xml":
                continue
            else:
                data_path.append(path.join(root, name))
                prefixs.append(pre)

    return data_path, prefixs


def rename_voc_dataset(anno_dir, jpg_dir, dst_dir=r"C:\dataset\license_plate\license_plate_detection"):
    # build files which will be used in VOC2007
    if not os.path.exists(os.path.join(dst_dir, "_Annotations")):
        os.mkdir(os.path.join(dst_dir, "_Annotations"))
    if not os.path.exists(os.path.join(dst_dir, "_JPEGImages")):
        os.mkdir(os.path.join(dst_dir, "_JPEGImages"))

    xml_paths, prefixs = get_path_prex(anno_dir)

    for i in range(len(xml_paths)):
        print(xml_paths[i])
        # rename and save the corresponding xml
        tree = read_xml(xml_paths[i])
        # save output xml, 000001.xml
        xml_fname = "{}.xml".format("%06d" % (i + 1))
        write_xml(tree, os.path.join(dst_dir, "_Annotations", xml_fname))

        # rename and save the corresponding image
        img_pre = prefixs[i] + ".jpg"
        root = jpg_dir
        img_path = path.join(root, img_pre)
        jpg_fname = '{}.jpg'.format("%06d" % (i + 1))
        img = Image.open(img_path).save(os.path.join(dst_dir, "_JPEGImages", jpg_fname))

# 程序功能：批量修改VOC数据集中xml标签文件的标签名称
def change_label_name(inputpath, src_name, dst_name):
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):  # 查找需要修改的名称
                    if (sku.text == src_name):  # 'license plate'为修改前的名称
                        sku.text = dst_name  # 'single-row license plate'为修改后的名称
                        # 写进原始的xml文件并避免原始xml中文字符乱码
                        tree.write(file, encoding='utf-8')
                    else:
                        pass
        else:
            pass


def check_wh_ratio(inputpath):
    for file in glob(os.path.join(inputpath, '*.xml')):
        # print(file)
        tree = ET.parse(file)
        root = tree.getroot()
        for object1 in root.findall('object'):
            if object1.find('name').text == 'single-row':
                for sku in object1.findall('bndbox'):  # 查找需要修改的名称
                    xmin = int(sku.find('xmin').text)
                    xmax = int(sku.find('xmax').text)
                    ymin = int(sku.find('ymin').text)
                    ymax = int(sku.find('ymax').text)
                    ratio = (xmax-xmin) / (ymax-ymin)
                    # print(ratio)

                    if ratio < 0.8:
                        print(file, xmin, xmax, ymin, ymax, ratio)
                        root.remove(object1)
                        tree.write(file, encoding='utf-8')


def remove_small_object(inputpath, min_height=12, min_width=47):
    for file in glob(os.path.join(inputpath, '*.xml')):
        tree = ET.parse(file)
        root = tree.getroot()

        org = root.findall('size')[0]
        img_w = int(org.find('width').text)
        img_h = int(org.find('height').text)

        for object1 in root.findall('object'):
            for sku in object1.findall('bndbox'):  # 查找需要修改的名称
                xmin = int(sku.find('xmin').text)
                xmax = int(sku.find('xmax').text)
                ymin = int(sku.find('ymin').text)
                ymax = int(sku.find('ymax').text)
                # ratio = (xmax-xmin) / (ymax-ymin)

                ratio = ((xmax-xmin)*(ymax-ymin)) / (img_w * img_h)
                if ratio*100 < 0.040:
                    print(file, ratio*100)

                # if ((ymax-ymin) < min_height) or ((xmax-xmin) < min_width):
                #     print((ymax-ymin), file)
                    root.remove(object1)
                    tree.write(file, encoding='utf-8')


def divide_dataset(root_path, train_target_dir, val_target_dir):
    total = glob(os.path.join(root_path, '*.jpg'))
    random.shuffle(total)
    print("Total image number: {}".format(len(total)))
    train_set = random.sample(total, int(len(total) * 0.8))
    val_set = list(set(total) - set(train_set))

    for idx, i in enumerate(train_set):
        base_name = os.path.basename(i)
        ex_basename = base_name.split('.')[0]
        print(i)
        # if idx > 100: break
        copyfile(i, os.path.join(train_target_dir, base_name))
        copyfile(i.replace('.jpg', '.xml'), os.path.join(train_target_dir, "{}.xml".format(ex_basename)))

    for idx, i in enumerate(val_set):
        base_name = os.path.basename(i)
        ex_basename = base_name.split('.')[0]
        print(i)
        # if idx > 100: break
        copyfile(i, os.path.join(val_target_dir, base_name))
        copyfile(i.replace('.jpg', '.xml'), os.path.join(val_target_dir, "{}.xml".format(ex_basename)))


if __name__ == '__main__':
    # xml_dir = r'D:\FMLD_annotations\test'
    # change_label_name(xml_dir, 'masked_face', 'masked')
    # change_label_name(xml_dir, 'unmasked_face', 'unmasked')
    # change_label_name(xml_dir, 'incorrectly_masked_face', 'incorrectly')

    # root_dir = r'C:\dataset\license_plate\license_plate_detection'
    # rename_voc_dataset(root_dir, jpg_dir, xml_dir)
    # # check_wh_ratio(xml_dir)
    # remove_small_object(xml_dir, min_height=6, min_width=24)

    original_path = r"D:\medical-masks-dataset_clean\images"
    train_target_dir = r"D:\medical-masks-dataset_clean\train"
    val_target_dir = r"D:\medical-masks-dataset_clean\test"

    Path(train_target_dir).mkdir(exist_ok=True, parents=True)
    Path(val_target_dir).mkdir(exist_ok=True, parents=True)

    divide_dataset(original_path, train_target_dir, val_target_dir)
