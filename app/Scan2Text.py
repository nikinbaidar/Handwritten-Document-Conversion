import pickle
import tensorflow as tf
import keras
import torch
import cv2
from PIL import Image
import os
import sys
import warnings
import numpy as np

from keras.layers import StringLookup
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.catalog import Metadata
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from ultralytics import YOLO
from autocorrect import Speller

root = os.path.join("/home/nikin/projects", "Handwritten-Document-Conversion")

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

model_paths = {
    'detectron2': os.path.join(root, 'MODELS/models'),
    'yolo': os.path.join(root, 'notebook/end_end_yolo/best100epoch.pt'),
    'crnn': os.path.join(
        root, 'notebook/end_end_yolo/handwriting_recognizer50.h5'),
}


def cv2_imshow(img) -> None:
    cv2.imshow("", img)
    while cv2.waitKey() != 27:
        continue
    cv2.destroyAllWindows()
    return None


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
        self.img = im
        self.model = model_paths['detectron2']
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(self.model, "config.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(self.model, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold
        cfg.DEVICE = 'cpu'
        metadata_path = os.path.join(self.model, "metadata.txt")
        with open(metadata_path, "r") as f:
            labels = [line.strip() for line in f]
        self.metadata = Metadata()
        self.metadata.set(thing_classes=labels)
        self.predictor = DefaultPredictor(cfg)

    def detect_layout(self):
        self.predictions = self.predictor(self.img)
        instances = self.predictions["instances"].to("cpu")
        return instances

    def y_sort_layout(self):
        instances = self.detect_layout()
        boxes = instances.pred_boxes.tensor
        y_coords = boxes[:, 1]
        sorted_indices = torch.argsort(y_coords)
        sorted_boxes = boxes[sorted_indices]
        return sorted_boxes

    def visualize_layout(self):
        instances = self.detect_layout()
        v = Visualizer(self.img[:, :, ::-1], metadata=self.metadata,
                       scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(instances)
        out_im = out.get_image()
        # cv2_imshow(out_im)
        return Image.fromarray(out_im)


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
        self.threshold = {'y': 20, 'iou': 0.50}
        self.char_labels = os.path.join(
            root, "notebook/end_end_yolo/characters_list.pkl")
        with open(self.char_labels, "rb") as f:
            ff_loaded = pickle.load(f)
            char_to_num = StringLookup(vocabulary=ff_loaded, mask_token=None)
            self.num_to_char = StringLookup(
                vocabulary=char_to_num.get_vocabulary(),
                mask_token=None, invert=True)

    def predict_word_boundary(self):
        predicted_word_boxes = self.yolo(self.image, verbose=False)[0].boxes
        coordinates = predicted_word_boxes.xyxy.tolist()
        confidence_scores = predicted_word_boxes.conf.tolist()

        bounding_boxes_y_sorted = sorted(
            ((tuple(box), float(conf))
             for box, conf in zip(coordinates, confidence_scores)),
            key=lambda x: x[0][1]
        )

        lines, new_line = [], [(bounding_boxes_y_sorted[0])]
        for box, conf in bounding_boxes_y_sorted[1:]:
            prev_box, _ = new_line[-1]
            # Compare y1 coordinates to decide if coordinates should be grouped
            # together i.e. coordinates belong to the same line.
            if abs(box[1] - prev_box[1]) <= self.threshold['y']:
                new_line.append((box, conf))
            else:
                lines.append(new_line)
                new_line = [(box, conf)]
        lines.append(new_line)

        filtered_lines = []  # Remove overlapping boxes

        for line in lines:
            line.sort(key=lambda x: x[0][0])
            pure_boxes = []

            for box, current_conf in line:
                # Check for overlaps with active boxes
                overlapping = False
                # Check if the current coordinate overlaps with any pure_boxes
                for (coordinate, conf) in pure_boxes:
                    iou_score = compute_iou(box, coordinate)
                    if iou_score > self.threshold['iou']:
                        overlapping = True
                        if current_conf > conf:
                            pure_boxes.remove((coordinate, conf))
                        else:
                            break

                if not overlapping:
                    pure_boxes.append((box, current_conf))

            pure_boxes = [list(map(int, box[0])) for box in pure_boxes]

            filtered_lines.append(pure_boxes)

        x_y_sorted_boundaries = filtered_lines

        self.clusters = x_y_sorted_boundaries

        return None

    def spellchecker(self, text):
        return self.spell(text)

    def apply_ocr(self, img, spellcheck=False):
        self.image = img
        final_prediction = ""
        self.predict_word_boundary()

        # Loop through the lines and save images
        for line in self.clusters:
            for word in line:
                x1, y1, x2, y2 = word
                roi = self.image[y1:y2, x1:x2]
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

        # Get the first decoded sequence and trim to the first 27 characters
        results = decoded[0][0][:27]

        # Remove padding (-1) and gather valid characters
        valid_results = tf.boolean_mask(results, tf.not_equal(results, -1))

        # Convert numerical results to characters and join into a string
        return tf.strings.reduce_join(self.num_to_char(valid_results)).numpy().decode("utf-8")


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def main():
    ocr = None
    image_path = os.path.join(root, 'notebook/end_end_yolo/eng_EU_319.jpg')
    im = cv2.imread(image_path)

    TestTwo = LayoutParser(im)
    boxes = TestTwo.y_sort_layout()

    if ocr:
        TestOne = TextExtractor()
        digitized_text = TestOne.apply_ocr(im)
        print(digitized_text)
    sys.exit()


if __name__ == "__main__":
    main()
