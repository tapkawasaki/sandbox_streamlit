import queue

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer


RTC_CONFIGURATION = RTCConfiguration(
    {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})
MEDIA_STREAM_CONSTRAINTS = {
    'video': True,
    'audio': False, }

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def detect_holistic(image):
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5) as pose:
        image.flags.writeable = False
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results


def plot_to_img(img, results):
    annotated_img = img.copy()
    mp_drawing.draw_landmarks(
        annotated_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return annotated_img


class VideoProcessor:

    def __init__(self) -> None:
        self.result_queue = queue.Queue()

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        results = detect_holistic(img)
        if results.pose_landmarks is not None:
            hands = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y]
            if hands[0] <= 0.1 or hands[0] >= 0.9:
                cv2.putText(img, 'hit!', (
                    int(hands[0]*img.shape[0]), int(hands[1]*img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 5, cv2.LINE_AA)
            elif hands[1] <= 0.1 or hands[1] >= 0.9:
                cv2.putText(img, 'hit!', (
                    int(hands[0]*img.shape[0]), int(hands[1]*img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 5, cv2.LINE_AA)
            elif hands[2] <= 0.1 or hands[2] >= 0.9:
                cv2.putText(img, 'hit!', (
                    int(hands[2]*img.shape[0]), int(hands[3]*img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 5, cv2.LINE_AA)
            elif hands[3] <= 0.1 or hands[3] >= 0.9:
                cv2.putText(img, 'hit!', (
                    int(hands[2]*img.shape[0]), int(hands[3]*img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 5, cv2.LINE_AA)
            self.result_queue.put(hands)
            annotated_img = plot_to_img(img, results)
        else:
            annotated_img = img
        img_dst = cv2.hconcat([img, annotated_img])
        return av.VideoFrame.from_ndarray(img_dst, format='bgr24')

if __name__ == '__main__':
    st.title('ボクシング')
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
            'autoPlay': True, },)
