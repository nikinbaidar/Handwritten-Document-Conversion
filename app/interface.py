#!/usr/bin/env python

import streamlit as st
import numpy as np
import os
from PIL import Image
from Scan2Text import TextExtractor, LayoutParser

import warnings
warnings.filterwarnings("ignore")


def getLayout(img):
    detector = LayoutParser(img)
    boxes, out_im = detector.y_sort_layout()
    text_regions = [img[top:bottom, left:right]
                    for left, top, right, bottom in boxes]
    return text_regions, out_im


def extractText(text_regions, spellcheck):
    for im_ in text_regions:
        output = extractor.apply_ocr(im_, spellcheck=spellcheck)
        output = output + "\n"
        with open(file_name, "a") as f:
            f.write(output)
    with open(file_name, "r") as f:
        file_contents = f.read()
    return file_contents


root = os.path.join("/home/nikin/projects", "Handwritten-Document-Conversion")

st.set_page_config(page_title="Handwritten Document Conversion",
                   page_icon=os.path.join(root, "app/favicon.ico"),
                   layout="wide")
st.title("Handwritten Document Conversion - 1")


# Some session states:

if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0

# Keep track of the previous uploaded file
if 'prev_uploaded_file' not in st.session_state:
    st.session_state.prev_uploaded_file = None

input = st.file_uploader("", type=['JPEG', 'PNG', 'TIFF', 'HEIF'])

if input is not None:
    if input != st.session_state.prev_uploaded_file:
        st.session_state.rotation_angle = 0
        st.session_state.prev_uploaded_file = input
    image = Image.open(input)
    file_name = f"{input.name.split('.')[0]}.txt"

    if os.path.exists(file_name):
        os.remove(file_name)

    # Rotation buttons
    _, col_a, col_b, col_c, _ = st.columns([8, 2, 2, 2, 30])
    with col_a:
        rotate_clockwise = st.button("↻")
    with col_b:
        rotate_counterclockwise = st.button("↺")
    # with col_c:
    #     pass

    rotate_angle = 90
    if rotate_clockwise:
        st.session_state.rotation_angle -= rotate_angle
    elif rotate_counterclockwise:
        st.session_state.rotation_angle += rotate_angle
    # elif rotate_ninety:
    #     st.session_state.rotation_angle += 90  # Rotate counterclockwise

    # Display Output
    col1, col2 = st.columns(2)

    input = image.rotate(st.session_state.rotation_angle, expand=True)
    with col1:
        st.image(input, caption="Input Image", use_column_width=True)
        spellcheck = st.checkbox("Spell check")
        process = st.button(label="Begin processing")

    if process:
        im = np.array(input)

        with col2:
            with st.spinner("Parsing Layout..."):
                text_regions, out_im = getLayout(im)
            st.image(out_im, caption="Layouts",
                     use_column_width=True)

        if text_regions:
            st.write("")
            # Avoid creating @tf.function repeatedly in a loop,
            extractor = TextExtractor()
            with st.spinner("OCR engine is running... this might take some time!"):
                file_contents = extractText(text_regions, spellcheck)
            st.markdown("##### Extracted Text:")
            st.code(file_contents, language='text')
            st.download_button(
                label="Save as .txt",
                data=file_contents.encode('utf-8'),
                file_name=file_name,
                mime="text/plain"
            )

        else:
            st.write("Sorry! our model could not detect any text regions.")
            print("Is the input correct?",
                  "Check i/p format, orientation, backgrounds and angle")


st.write("")
st.markdown("Authors: [nikin](https://github.com/nikinbaidar), "
            "[nimesh](https://github.com/nikinbaidar)")
