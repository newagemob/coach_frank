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


def predict(video_path):
    # frame is a string representing the videos location in /tmp directory. the video is in mp4 format
    video = cv2.VideoCapture(video_path) # open the video
    # read the first frame
    success, frame = video.read()
    
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # resize the frame to the input shape of the model
    gray = cv2.resize(gray, (2048, SEQUENCE_LENGTH))
    
    # normalize the pixel values
    gray = gray / 255
    
    # add an extra dimension to the frame for the model
    gray = np.expand_dims(gray, axis=0)
    
    # predict the trick using the model
    prediction = model.predict(gray)

    # get the index of the highest probability
    index = np.argmax(prediction)
    
    label_percentage = prediction[0][index]

    # get the label of the predicted trick
    label = labels[index]

    # print the predicted trick
    print("Predicted trick: " + label)
    print("Confidence: " + str(label_percentage))

    # return the predicted trick
    return label
    
    
iface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs,
                     capture_session=True, live=True, allow_flagging=False)

iface.queue()

iface.launch(share=True)
