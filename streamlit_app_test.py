# -*- coding: utf-8 -*-

import mediapipe as mp
import cv2

import av
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer, WebRtcMode, RTCConfiguration)

RTC_CONFIGURATION = RTCConfiguration(
    {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]}
)

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
    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        results = detect_face(img)
        if results.multi_face_landmarks is not None:
            annotated_img = plot_to_img(img, results)
        else:
            annotated_img = img
        img_dst = cv2.hconcat([img, annotated_img])
        return av.VideoFrame.from_ndarray(img_dst, format='bgr24')

st.title('WebCam App')
webrtc_streamer(key='sample',
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    mode=WebRtcMode.SENDRECV,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},)