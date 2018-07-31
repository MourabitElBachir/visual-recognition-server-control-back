import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob

from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

from multiprocessing.dummy import Pool as ThreadPool

# Windows dependencies
# - Python 2.7.6: http://www.python.org/download/
# - OpenCV: http://opencv.org/
# - Numpy -- get numpy from here because the official builds don't support x64:
#   http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy

# Mac Dependencies
# - brew install python
# - pip install numpy
# - brew tap homebrew/science
# - brew install opencv

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    out = cv2.imwrite(os.path.join('image_upload', 'capture.jpg'), frame)

    cv2.imshow('Buttons Detection', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


def get_correct_path(files):

    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  files)


def object_detection_runner(filename):

    UPLOAD_FOLDER = 'image_upload'
    OUTPUT_FOLDER = 'image_output'

    MAX_NUMBER_OF_BOXES = 10
    MINIMUM_CONFIDENCE = 0.9

    PATH_TO_LABELS = get_correct_path('annotations/label_map.pbtxt')
    PATH_TO_TEST_IMAGES_DIR = UPLOAD_FOLDER

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize,
                                                                use_display_name=True)
    CATEGORY_INDEX = label_map_util.create_category_index(categories)

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    MODEL_NAME = get_correct_path('graphs')
    PATH_TO_CKPT = MODEL_NAME + '/ssd_mobilenet_v1.pb'

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def detect_objects(image_path):
        import ntpath
        head, tail = ntpath.split(image_path)
        image_name = tail or ntpath.basename(head)
        print(image_name)

        image = Image.open(image_path)
        (im_width, im_height) = image.size
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            CATEGORY_INDEX,
            min_score_thresh=MINIMUM_CONFIDENCE,
            use_normalized_coordinates=True,
            line_thickness=8)
        fig = plt.figure()
        dpi = 100
        im_width_inches = im_width / dpi
        im_height_inches = im_height / dpi
        fig.set_size_inches(im_width_inches, im_height_inches)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.imshow(image_np, aspect='auto')
        plt.savefig(os.path.join(OUTPUT_FOLDER, image_name), dpi=62)
        plt.close(fig)

    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image-{}.jpg'.format(i)) for i in range(1, 4) ]
    TEST_IMAGE_PATH = os.path.join(PATH_TO_TEST_IMAGES_DIR, filename)

    # Load model into memory
    print('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print('detecting...')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            detect_objects(TEST_IMAGE_PATH)
