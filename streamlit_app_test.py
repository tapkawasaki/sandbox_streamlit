# -*- coding: utf-8 -*-

import mediapipe as mp
from deepface import DeepFace

import pandas as pd
import cv2
import queue

import av
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer, WebRtcMode, RTCConfiguration)

RTC_CONFIGURATION = RTCConfiguration(
    {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})
MEDIA_STREAM_CONSTRAINTS = {'video': {'frameRate': { 'ideal': 5, 'max': 10 }},
                            'audio': False,}

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def detect_face(img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img)
    return results

def detect_emotion(img, backend='ssd'):
    try:
        emotion = DeepFace.analyze(img, actions=['emotion'], detector_backend=backend)
    except:
        emotion = None
    return emotion

def plot_to_img(img, results):
    annotated_img = img.copy()
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    return annotated_img

class VideoProcessor:
    
    def __init__(self) -> None:
        self.result_queue = queue.Queue()
    
    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        results = detect_face(img)
        emotion = detect_emotion(img)
        self.result_queue.put(emotion)
        if results.multi_face_landmarks is not None:
            annotated_img = plot_to_img(img, results)
        else:
            annotated_img = img
        img_dst = cv2.hconcat([img, annotated_img])
        return av.VideoFrame.from_ndarray(img_dst, format='bgr24')

if __name__ == '__main__':
    st.title('WebCam App')
    webrtc_ctx = webrtc_streamer(
        key='sample',
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        mode=WebRtcMode.SENDRECV,
        async_processing=True,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        video_html_attrs={
            'style': {'width': '75%', 'margin': '0 auto', 'border': '1px black solid'},
            'controls': False,
            'autoPlay': True,},)

    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            df_emotion = pd.DataFrame(
                index=[],
                columns=['emotion', 'probability'])
            if webrtc_ctx.video_processor:
                try:
                    emotion = webrtc_ctx.video_processor.result_queue.get(
                        timeout=1.0
                    )
                    if emotion:
                        df_emotion = pd.DataFrame({
                            'emotion':emotion['emotion'].keys(),
                            'probability':emotion['emotion'].values()})
                    else:
                        df_emotion = pd.DataFrame({
                            'emotion':['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                            'probability':[0]*7})
                except queue.Empty:
                    pass
                labels_placeholder.table(df_emotion)
            else:
                break
