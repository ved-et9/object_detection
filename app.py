from yolo_predicting import YOLO_Pred
import cv2
import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import numpy as np
import tempfile

from yolo_predicting import YOLO_Pred

yolo=YOLO_Pred('./best.onnx','data.yaml')


def main():
    st.header('𝒪𝐵𝒥𝐸𝒞𝒯 𝒟𝐸𝒯𝐸𝒞𝒯𝐼𝒪𝒩 𝒰𝒮𝐼𝒩𝒢 𝒴𝒪𝐿𝒪𝒱5')
    image = Image.open('images/62321f913bcc1326c3d098e6_portada.png')

    st.image(image, caption='Sunrise by the mountains')

    st.sidebar.header('Menu')

    selected_box = st.sidebar.selectbox(
        '--------------------------------',
        ('Welcome', 'About Yolo' ,'Model weights and curve','Object Detection Live Demo')
    )





    if selected_box == 'Object Detection Live Demo':
        option = st.selectbox(
            "What would you like to upload?",
            ("Image", "Video"),
            index=None,
            placeholder="Select contact method...",
        )
        if option == 'Image':

            image = st.file_uploader("Upload an image", type=["JPEG", "JPG", "PNG"])
            if image is not None:
                # Load image using OpenCV
                img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
                img_pred = yolo.predictions(img)
                st.image(img_pred, caption='Sunrise by the mountains')
        elif option == 'Video':
            f = st.file_uploader("Upload file")

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            cap = cv2.VideoCapture(tfile.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if ret == False:
                    print('unable to read video')
                    break

                pred_image = yolo.predictions(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(gray)
    elif selected_box == 'Model weights and curve':
        st.subheader('These following graphs were obtained')
        st.image('images/labels.jpg', caption='Labels', use_column_width=True,)
        st.image('images/labels_correlogram.jpg', caption='Label_Correlogram', use_column_width=True)
        st.image('images/P_curve.png', caption='P_curve', use_column_width=True, )
        st.image('images/R_curve.png', caption='R_curve', use_column_width=True, )

        st.image('images/results.png', caption='results', use_column_width=True, )
        st.image('images/train_batch2.jpg', caption='train_batch', use_column_width=True, )
        st.image('images/val_batch2_labels.jpg', caption='val_batch2_labels', use_column_width=True, )
    elif selected_box == 'About Yolo':
        st.text('''You Only Look Once (YOLO) proposes using an end-to-end neural network that
    makes predictions of bounding boxes and class probabilities all at once.
    It differs from the approach taken by previous object detection algorithms, 
    which repurposed classifiers to perform detection.
    Following a fundamentally different approach to object detection, 
    YOLO achieved state-of-the-art results, 
    beating other real-time object detection algorithms by a large margin.
    While algorithms like Faster RCNN work by detecting possible
    regions of interest using the Region Proposal Network and then performing
    recognition on those regions separately,YOLO performs all of its predictions
    with the help of a single fully connected layer.Methods that use Region 
    Proposal Networks perform multiple iterations for the same image, 
    while YOLO gets away with a single iteration.Several new versions of the same 
    model have been proposed since the initial release of YOLO in 2015, 
    each building on and improving its predecessor. 
    Here's a timeline showcasing YOLO's development in recent years.''')
        st.subheader('YOLO Architecture')
        st.image('images/63c697fd4ef3d83d2e35a8c2_YOLO architecture-min.jpg')
        st.subheader('YOLOv5')
        st.text('''YOLO v5 was introduced in 2020 by the same team that developed
                the original YOLO algorithm as an open-source project and is 
                maintained by Ultralytics. YOLO v5 builds upon the success of 
                previous versions and adds several new features and improvements.
                Unlike YOLO, YOLO v5 uses a more complex architecture called 
                EfficientDet (architecture shown below), based on the EfficientNet 
                network architecture. Using a more complex architecture in 
                YOLO v5 allows it to achieve higher accuracy and better 
                generalization to a wider range 
                of object categories.''')
        st.image('images/v5.jpg')




if __name__ == "__main__":
    main()