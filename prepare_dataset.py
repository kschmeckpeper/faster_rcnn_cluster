import json
import os
import numpy as np

def gen_header(image):
    filename = image['file_name']
    height = image['height']
    width = image['width']
    header= """<annotation>
            	<folder>VOC2007</folder>
            	<filename>FILENAME</filename>
            	<source>
            		<database>The VOC2007 Database</database>
            		<annotation>PASCAL VOC2007</annotation>
            	</source>
            	<size>
            		<width>WIDTH</width>
            		<height>HEIGHT</height>
            		<depth>3</depth>
            	</size>
            	<segmented>0</segmented>
            """
    header = header.replace('FILENAME', filename)
    header = header.replace('WIDTH', str(width))
    header = header.replace('HEIGHT', str(height))
    return header

def gen_object(bbox, object_class):
    header = """<object>
		   <name>OBJECT_CLASS</name>
		   <pose>Right</pose>
		   <truncated>0</truncated>
		   <difficult>0</difficult>
		   <bndbox>
	               <xmin>XMIN</xmin>
	               <ymin>YMIN</ymin>
	               <xmax>XMAX</xmax>
	               <ymax>YMAX</ymax>
		   </bndbox>
             </object>
             """
    header = header.replace('OBJECT_CLASS', object_class)
    header = header.replace('XMIN', str(bbox[0]))
    header = header.replace('YMIN', str(bbox[1]))
    header = header.replace('XMAX', str(bbox[2]))
    header = header.replace('YMAX', str(bbox[3]))
    return header

destination = '/NAS/data/rcta/VOCdevkitall/VOC2007'
# Make folders that replicate the VOC directory structure
annotations_folder = os.path.join(destination, 'Annotations')
image_sets_folder = os.path.join(destination, 'ImageSets', 'Main')
jpeg_images_folder = os.path.join(destination, 'JPEGImages')
segmentation_class_folder = os.path.join(destination, 'SegmentationClass')
segmentation_object_folder = os.path.join(destination, 'SegmentationObject')
if not os.path.exists(annotations_folder):
    os.makedirs(annotations_folder)
if not os.path.exists(image_sets_folder):
    os.makedirs(image_sets_folder)
if not os.path.exists(jpeg_images_folder):
    os.makedirs(jpeg_images_folder)
if not os.path.exists(segmentation_class_folder):
    os.makedirs(segmentation_class_folder)
if not os.path.exists(segmentation_object_folder):
    os.makedirs(segmentation_object_folder)

with open('/NAS/data/rcta/lejeune_figure8/object_bounding_boxes/annotations/lejeune_data-2019-03-27.json', 'r') as f:
    data = json.load(f)

categories = data['categories']
images = data['images']
annotations = data['annotations']

img_cnt = 0
xml_files = [gen_header(images[i]) for i in range(len(images))]
for i in range(len(annotations)):
    image_id = annotations[i]['image_id'] - 1
    obj_class = categories[annotations[i]['category_id'] - 1]['name'].lower()
    bbox = annotations[i]['bbox']
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    xml_files[image_id] += gen_object(bbox, obj_class)
xml_files = [f + '</annotation>' for f in xml_files]

filenames = [i['file_name'] for i in images]
for content, name in zip(xml_files, filenames):
    name = name[:-3] + 'xml'
    with open(os.path.join(annotations_folder, name), 'w') as f:
        f.write(content)

perm = np.random.permutation(range(len(filenames)))
filenames = np.array(filenames)
total = len(perm)
perm_train = perm[:int(0.9*total)]
perm_test = perm[:int(0.9*total):]

trainval = filenames[perm_train]
test = filenames[perm_test]

with open(os.path.join(image_sets_folder, 'trainval.txt'), 'w') as f:
    f.writelines([t[:-4] + '\n' for t in trainval])
with open(os.path.join(image_sets_folder, 'test.txt'), 'w') as f:
    f.writelines([t[:-4] + '\n' for t in test])
