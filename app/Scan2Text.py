import pickle
import tensorflow as tf
import keras
import torch
import cv2
from PIL import Image
import os
import sys
import numpy as np
import random

from keras.layers import StringLookup

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from ultralytics import YOLO
from autocorrect import Speller


import warnings
warnings.filterwarnings("ignore")

root = os.path.join("/home/nikin/projects", "Handwritten-Document-Conversion")
np.random.seed(42)
tf.random.set_seed(42)

model_paths = {
    'detectron2': os.path.join(root, 'MODELS/models'),
    'yolo': os.path.join(root, 'notebook/end_end_yolo/bestv10.pt'),
    'crnn': os.path.join(
        root, 'notebook/end_end_yolo/handwriting_recognizer50.h5'),
}


def scale_img(im, scale=1.0):
    if not isinstance(im, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray got {type(im)}")

    height, width = im.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    im = cv2.resize(im, (new_width, new_height))
    return im


def cv2_imshow(im, scale=None):
    if not isinstance(im, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray got {type(im)}")

    if scale is not None:
        im = scale_img(im, scale=scale)

    cv2.imshow("", im)
    while True:
        if cv2.waitKey() == 27:
            cv2.destroyAllWindows()
            break


def random_color():
    r = random.randint(0, 256)
    g = random.randint(0, 128)
    b = random.randint(0, 256)
    return (r, g, b)


class CTCLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype=tf.int64)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int64)
        label_length = tf.cast(tf.shape(y_true)[1], dtype=tf.int64)

        input_length = input_length * \
            tf.ones(shape=(batch_len, 1), dtype=tf.int64)
        label_length = label_length * \
            tf.ones(shape=(batch_len, 1), dtype=tf.int64)
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


class LayoutParser():
    def __init__(self, im, detection_threshold=0.85):
        """
        Initializes the LayoutParser instance for layout detection in document
        images.

        Args:
            * im (numpy.ndarray): The input image on which layout detection
            will be performed.
            * detection_threshold (float, optional): The score threshold for
            filtering predictions. (default 0.85). Predictions with scores
            below this threshold will be discarded.

        Attributes:
            * img (numpy.ndarray): The input image.
            * model (str): Path to the Detectron2 model directory.
            * predictor (DefaultPredictor): The predictor object configured
            with the model for making predictions on the input images.
        """
        self.img = im
        self.model = model_paths['detectron2']
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(self.model, "config.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(self.model, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold
        cfg.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(cfg)

    def detect_layout(self):
        """
        Detects layout instances in the input image using the configured
        predictor.

        This method processes the input image and generates predictions for
        layout instances. The predictions include bounding boxes, class labels,
        and scores for each detected instance. The results are transferred to
        the CPU for further processing.

        Returns:
            instances (Instances): An object containing detected layout
            instances, including their bounding boxes, class labels, and
            associated scores. The instances are represented on the CPU for
            easier manipulation.
        """
        predictions = self.predictor(self.img)
        instances = predictions["instances"].to("cpu")
        return instances

    def y_sort_layout(self):
        """
        Sorts the detected layout instances by their Y-coordinates and filters
        out boxes based on their area.

        This method performs the following steps:
        1. Detects layout instances from the image.
        2. Extracts the bounding boxes of the detected instances.
        3. Sorts the bounding boxes based on their Y-coordinates.
        4. Filters the sorted boxes by area to remove irrelevant boxes.
        5. Draws rectangles around the filtered boxes on the original image.

        Returns:
            tuple: A tuple containing:
                - pure_boxes (list): A list of filtered bounding boxes, each
                represented as a list of coordinates [x1, y1, x2, y2].
                - out_im (PIL.Image): The output image with rectangles drawn
                around the filtered boxes.
        """
        instances = self.detect_layout()
        boxes = instances.pred_boxes.tensor
        y_coords = boxes[:, 1]
        sorted_indices = torch.argsort(y_coords)
        sorted_boxes = boxes[sorted_indices]
        pure_boxes = filter_by_area(sorted_boxes.tolist())
        for box in pure_boxes:
            x1, y1, x2, y2 = box
            out_im = cv2.rectangle(self.img, (x1, y1), (x2, y2),
                                   color=random_color(), thickness=2)
        out_im = Image.fromarray(self.img)
        return pure_boxes, out_im


class TextExtractor():
    def __init__(self, yolo=None, crnn=None):
        if yolo is None:
            yolo = model_paths['yolo']

        if crnn is None:
            crnn = model_paths['crnn']

        self.yolo = YOLO(yolo)
        self.spell = Speller(lang="en", only_replacements=True)
        # custom_objects =
        with keras.utils.custom_object_scope({'CTCLayer': CTCLayer}):
            self.crnn = keras.models.load_model(crnn)
            self.ocr_agent = keras.models.Model(
                self.crnn.get_layer(name="image").input,
                self.crnn.get_layer(name="dense2").output)
        self.threshold = {'y': 20, 'containment_ratio': 0.50}
        self.char_labels = os.path.join(
            root, "notebook/end_end_yolo/characters_list.pkl")
        with open(self.char_labels, "rb") as f:
            ff_loaded = pickle.load(f)
            char_to_num = StringLookup(vocabulary=ff_loaded, mask_token=None)
            self.num_to_char = StringLookup(
                vocabulary=char_to_num.get_vocabulary(),
                mask_token=None, invert=True)

    def predict_word_boundary(self):
        predicted_word_boxes = self.yolo(self.img, verbose=False)[0].boxes
        coordinates = predicted_word_boxes.xyxy.tolist()

        bounding_boxes_y_sorted = sorted(
            [tuple(box) for box in coordinates], key=lambda x: x[1])
        average_height = sum(bx[3] - bx[1] for bx in bounding_boxes_y_sorted) \
            / len(bounding_boxes_y_sorted)

        lines, new_line = [], [(bounding_boxes_y_sorted[0])]

        for box in bounding_boxes_y_sorted[1:]:
            prev_box = new_line[-1]
            # Compare y1 coordinates to decide if coordinates should be grouped
            # together i.e. coordinates belong to the same line.
            if abs(box[1] - prev_box[1]) <= average_height * 0.51:
                new_line.append(box)
            else:
                lines.append(new_line)
                new_line = [box]
        lines.append(new_line)

        x_y_sorted_boundaries = []

        for line in lines:
            line.sort(key=lambda x: x[0])
            x_y_sorted_boundaries.append(filter_by_area(line))

        self.clusters = x_y_sorted_boundaries

        return None

    def spellchecker(self, text):
        return self.spell(text)

    def preprocess_image(self, word):
        if isinstance(word, np.ndarray):
            word = cv2.cvtColor(word, cv2.COLOR_RGB2GRAY)
            normalized_word = word / 255.0
            image = tf.convert_to_tensor(
                np.expand_dims(normalized_word, axis=-1))
            image = tf.cast(image, tf.float32)
            # Resize image
            new_size = tuple(reversed((128, 32)))
            resized_image = tf.image.resize(
                image, size=new_size, preserve_aspect_ratio=True)

            x_off = new_size[1] - tf.shape(resized_image)[1]
            y_off = new_size[0] - tf.shape(resized_image)[0]

            right = x_off // 2
            bottom = y_off // 2
            left = right + x_off % 2
            top = bottom + y_off % 2

            padded_image = tf.pad(resized_image, paddings=[
                [top, bottom], [left, right], [0, 0], ],)

            image = tf.transpose(padded_image, perm=[1, 0, 2])
            image = tf.image.flip_left_right(image)
            image = tf.expand_dims(image, axis=0)
            return image

    def decode_ctc_prediction(self, pred):
        input_len = tf.fill([1], tf.shape(pred)[1])
        decoded, _ = tf.keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True)
        results = decoded[0][0][:50]

        # Remove padding (-1) and gather valid characters
        valid_results = tf.boolean_mask(results, tf.not_equal(results, -1))

        # Convert numerical results to characters and join into a string
        return tf.strings.reduce_join(self.num_to_char(valid_results)).numpy().decode("utf-8")

    def apply_padding(self):
        padded_image = cv2.copyMakeBorder(
            self.img, 100, 100, 100, 100,
            cv2.BORDER_CONSTANT, value=(255, 255, 255))
        self.img = padded_image

    def apply_ocr(self, img, spellcheck=False):
        self.img = img
        self.apply_padding()
        self.predict_word_boundary()
        final_prediction = ""

        # Loop through the lines and save images
        for line in self.clusters:
            for word in line:
                x1, y1, x2, y2 = word
                roi = self.img[y1:y2, x1:x2]
                preprocessed_roi = self.preprocess_image(roi)
                prediction = self.ocr_agent.predict(
                    preprocessed_roi, verbose='0')
                predicted_text = self.decode_ctc_prediction(prediction)
                if spellcheck:
                    corrected_text = self.spellchecker(predicted_text)
                    final_prediction = final_prediction + corrected_text + " "
                else:
                    final_prediction = final_prediction + predicted_text + " "

            final_prediction = final_prediction + "\n"
        return final_prediction


def compute_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def compute_containment_ratio(boxA, boxB):
    # Calculate intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate width and height of the intersection area
    inter_width = max(0, xB - xA + 1)
    inter_height = max(0, yB - yA + 1)

    # If no intersection, return 0 early
    if inter_width == 0 or inter_height == 0:
        return 0

    # Compute the intersection area
    inter_area = inter_width * inter_height

    boxA_area = compute_area(boxA)
    boxB_area = compute_area(boxB)
    smaller_area = min(boxA_area, boxB_area)

    return inter_area / smaller_area if smaller_area > 0 else 0


def filter_by_area(group, threshold=0.8):

    pure_boxes = []

    for box in group:
        keep_box = True
        box_area = compute_area(box)

        for other_box in group:
            if other_box == box:
                continue

            containment_ratio = compute_containment_ratio(
                box, other_box)

            if containment_ratio > threshold:
                other_box_area = compute_area(other_box)

                if box_area < other_box_area:
                    keep_box = False
                    break

        if keep_box:
            pure_boxes.append(list(map(int, box)))
    return pure_boxes


def main():
    ocr = True
    layout_detection = True
    image_path = os.path.join(root, 'notebook/end_end_yolo/eng_AS_077.jpg')
    image_path = '/home/nikin/WhatsApp Image 2024-09-27 at 11.05.43 PM.jpeg'
    im = cv2.imread(image_path)

    if layout_detection:
        TestTwo = LayoutParser(im)
        boxes, img = TestTwo.y_sort_layout()
        img = np.array(img)
        cv2_imshow(img, scale=0.5)

    if ocr:
        TestOne = TextExtractor()
        digitized_text = TestOne.apply_ocr(im)
        print(digitized_text)


if __name__ == "__main__":
    main()
