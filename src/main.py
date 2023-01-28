import tensorflow as tf
import numpy as np
import gradio as gr
import cv2
import os

SEQUENCE_LENGTH = 40

model = tf.keras.models.load_model('trick_classifier.h5')

labels = ['Ollie', 'Kickflip', 'Shuvit']

inputs = gr.components.Video(
    source='webcam', format='mov', label='Coach Frank is watching you...', type='file', capture_session=True)

outputs = gr.outputs.Textbox(label='Coach Frank says:')


def predict(frame):
    # frame is a string representing the videos location in /tmp directory. the video is in mp4 format
    video = cv2.VideoCapture(frame)  # open the video
    
    # get the number of frames in the video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every_frame = max(1, num_frames // SEQUENCE_LENGTH)
    current_frame = 0
    max_images = SEQUENCE_LENGTH

    while True:
        success, frame = video.read()
        if not success:
            break

        if current_frame % sample_every_frame == 0:
            # OPENCV reads in BGR, tensorflow expects RGB so we invert the order
            frame = frame[:, :, ::-1]
            # shape data to match model [None, 40, 2048]
            img = tf.reshape(frame, (-1, 3))
            img = tf.image.resize(img, (SEQUENCE_LENGTH, 2048))
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            max_images -= 1
        yield img, frame
        current_frame += 1

        if max_images == 0:
            break
    
    # get the prediction of the current video
    prediction = model.predict(img)
    print(f'👉 Prediction: {prediction}')
    
    # get the index of the highest probability
    index = np.argmax(prediction)
    
    # return the label of the highest probability
    # return labels[index]
  

iface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs,
                     capture_session=True, live=True, allow_flagging=False)

iface.queue()

iface.launch(share=True)
