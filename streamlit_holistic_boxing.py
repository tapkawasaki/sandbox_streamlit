import queue

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

RTC_CONFIGURATION = RTCConfiguration(
    {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})
MEDIA_STREAM_CONSTRAINTS = {
    'video': {'frameRate': {'ideal': 10, 'max': 15}},
    'audio': False,
}

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def detect_holistic(image):
    model_complexity = 1
    with mp_pose.Pose(
            static_image_mode=False, model_complexity=model_complexity, min_detection_confidence=0.5) as pose:
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
        self.hit_threshould = 0.15
        img = frame.to_ndarray(format='bgr24')
        annotated_img = img.copy()
        results = detect_holistic(img)
        if results.pose_landmarks is not None:
            hands = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].z,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].z]
            if hands[0] <= self.hit_threshould or hands[0] >= 1-self.hit_threshould or\
                    hands[1] <= self.hit_threshould or hands[1] >= 1-self.hit_threshould:
                cv2.putText(annotated_img, 'Hit!', (
                    int(hands[0]*annotated_img.shape[0]), int(hands[1]*annotated_img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5, cv2.LINE_AA)
            if hands[3] <= self.hit_threshould or hands[3] >= 1-self.hit_threshould or\
                    hands[4] <= self.hit_threshould or hands[4] >= 1-self.hit_threshould:
                cv2.putText(annotated_img, 'Hit!', (
                    int(hands[3]*annotated_img.shape[0]), int(hands[4]*annotated_img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5, cv2.LINE_AA)
            self.result_queue.put(hands)
            annotated_img = plot_to_img(annotated_img, results)
        img_dst = annotated_img#cv2.hconcat([img, annotated_img])
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
            'style': {'width': '100%', 'margin': '0 auto', 'border': '1px black solid'},
            'controls': False,
            'autoPlay': True, },)

    if webrtc_ctx.state.playing:
        temp = []
        labels_placeholder = st.empty()
        while True:
            df = pd.DataFrame(
                index=['x', 'y', 'z', 'speed'],
                columns=['Left', 'Right'])
            if webrtc_ctx.video_processor:
                try:
                    hands = webrtc_ctx.video_processor.result_queue.get(
                        timeout=1.0
                    )
                    temp.append(hands)
                    if len(temp) >= 3:
                        temp.pop(0)
                    if len(temp) == 1:
                        left_0 = np.array((temp[0][0], temp[0][1]))
                        left_1 = left_0
                        right_0 = np.array((temp[0][3], temp[0][4]))
                        right_1 = right_0
                    else:
                        left_0 = np.array((temp[0][0], temp[0][1]))
                        left_1 = np.array((temp[1][0], temp[1][1]))
                        right_0 = np.array((temp[0][3], temp[0][4]))
                        right_1 = np.array((temp[1][3], temp[1][4]))
                    dist_left = np.linalg.norm(left_0-left_1)
                    dist_right = np.linalg.norm(right_0-right_1)

                    if hands:
                        df['Left'] = hands[:3] + [dist_left]
                        df['Right'] = hands[3:] + [dist_right]

                    else:
                        df['Left'] = [0]*4
                        df['Right'] = [0]*4
                except queue.Empty:
                    pass
                labels_placeholder.table(df)
            else:
                break
