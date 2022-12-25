import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = [('', 'train'), ('', 'val'), ('', 'test')]

classes = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_back',
              '1_dashing', '1_downSlash', '1_idle', '1_jump', '1_knocked','1_parry', '1_upSlash',
              '2_parry', '2_u', '2_defeated', '2_back', '2_i', '2_dashing', '2_jump', '2_skill3', '2_j', '2_skill2', '2_l', '2_skill1', '2_o', '2_idle', '2_k']

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('/data/lrl/SAMURAI_mult%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('/data/lrl/SAMURAI_mult%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('/data/lrl/SAMURAI_mult%s/labels/'%(year)):
        os.makedirs('/data/lrl/SAMURAI_mult%s/labels/'%(year))
    image_ids = open('/data/lrl/SAMURAI_mult%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('/data/lrl/SAMURAI_mult%s/images/%s.jpg\n'%(year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

