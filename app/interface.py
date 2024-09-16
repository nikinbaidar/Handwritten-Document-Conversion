#!/usr/bin/env python

# <TODO>
# 1. Docstrings for all functions and classes
# 1. Create a pipeline


################
# LOAD MOUDLES #
################

import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import json
import random
import pytesseract
import warnings

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.catalog import Metadata
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from spellchecker import SpellChecker

os.system("clear")
warnings.filterwarnings("ignore")

class TextExtractor:
    def __init__(self, img):
        self.img = np.array(img)
        self.images = self.preprocessImage()

    def extract_text(self, img):
        # Extract text from the image using pytesseract
        return pytesseract.image_to_string(img)


    def preprocessImage(self):
        img_grayscaled = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_grayscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img_reduced_noise = cv2.GaussianBlur(img_binary, (3, 3), 0)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # img_dilated = cv2.dilate(img_reduced_noise, kernel, iterations=1)
        # img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

        return {
            # 'original' : self.img,
            # 'grayscale' : img_grayscaled,
            'binary' : img_binary,
            # 'no_noise' : img_reduced_noise,
            # 'dilated' : img_dilated,
            # 'eroded' : img_eroded
        }

    def correct_text(self, text):
        spell = SpellChecker()
        lines = text.splitlines()
        lines = [ line for line in lines if line ]
        corrected_lines = []
        for line in lines:
            words = line.split()
            corrected_sentence = []
            for word in words:
                if word not in spell:
                    corrected_word = spell.correction(word)
                    corrected_sentence.append(corrected_word)
                else:
                    corrected_sentence.append(word)
            corrected_sentence = [item for item in corrected_sentence if item is not None]
            corrected_lines.append(" ".join(corrected_sentence))

        corrected_text = "\n".join(corrected_lines) + "\n"
        return corrected_text


    def get_output(self):
      predicted = [ self.extract_text(im) for _, im in self.images.items() ]
      corrected = [ self.correct_text(text) for text in predicted ]
      return {
           'predicted': predicted,
           'corrected': corrected,
      }


MODEL_PATH = "/home/nikin/projects/Handwritten-Document-Conversion/MODELS/models/"
DETECTION_THRESHOLD = 0.90

cfg = get_cfg()
cfg.merge_from_file(os.path.join(MODEL_PATH, "config.yaml"))
cfg.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = os.path.join(MODEL_PATH, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_THRESHOLD
predictor = DefaultPredictor(cfg)


# Define the labels for prediction
metadata_path = os.path.join(MODEL_PATH, "metadata.txt")
with open(metadata_path, "r") as f:
  labels = [ line.strip() for line in f ]

metadata = Metadata()
metadata.set(thing_classes = labels)

##################
# UI BEGINS HERE #
##################

st.set_page_config(page_title="Handwritten Document Conversion", page_icon="./favicon.ico", layout="wide")

st.title("Handwritten Document Conversion - 1")
st.subheader("Nikin Baidar & Nimesh Gopal Pradhan")

uploaded_file = st.file_uploader("Upload a file...",
        type=['JPEG', 'PNG', 'TIFF', 'HEIF'])

if uploaded_file is not None:
    st.write(f"File '{uploaded_file.name}' uploaded successfully!")
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    output_filename = uploaded_file.name.split(".")[0]
    filename = f'{output_filename}.txt'

    ####################
    # PARSE THE LAYOUT #
    ####################

    im = np.array(image)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_im = Image.fromarray(out.get_image())
    with col2:
        st.image(out_im, caption=f"Detected Layout with {float(DETECTION_THRESHOLD) * 100}% threshold", use_column_width=True)

    for _ in range(2):
        st.write("")

    boxes = outputs['instances'].pred_boxes.tensor
    y_coords = boxes[:, 1]
    sorted_indices = torch.argsort(y_coords)
    sorted_boxes = boxes[sorted_indices]

    cropped_images = []
    for bbox in sorted_boxes:
      box = bbox.tolist()
      left, top, right, bottom = map(int, box)
      im_ = im[top:bottom, left:right]
      cropped_images.append(im_)

    if cropped_images:
        tags = ['original', 'grayscale', 'binary', 'no_noise', 'dilated', 'eroded' ]
        index = tags.index('binary')
        for im in cropped_images:
            extractor = TextExtractor(im)
            output = extractor.get_output()
            text = output['corrected'][0]
            with open(filename, "a") as f:
                f.write(text)

        # Display Output
        with open(filename, "r") as f:
            file_contents = f.read()

        st.write("")
        st.header("Extracted Text:")
        st.code(file_contents, language='text')


        # Provide a download link for the file
        with open(filename, "rb") as f:
            st.download_button(
                    label="Save as .txt",
                    data=f,
                    file_name=filename,
                    mime="text/plain"
            )

        if os.path.exists(filename):
            os.remove(filename)
    else:
        print("Is the input correct?", "Check format, orientation, backgrounds and angle")
        st.write("Sorry! our model could not detect any text regions. Please try another image.")
