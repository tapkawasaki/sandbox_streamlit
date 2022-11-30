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


@st.cache(ttl=60*1)
def detect_pose(image: np.array):
    model_complexity = 0
    with mp_pose.Pose(
            static_image_mode=False, model_complexity=model_complexity,
            min_detection_confidence=0.5) as pose:
        image.flags.writeable = False
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print(results)
    return results


def plot_to_img(img: np.array, results) -> np.array:
    annotated_img = img.copy()
    mp_drawing.draw_landmarks(
        annotated_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return annotated_img


def detect_hit(hands: list, hit_point: list, hit_radius: int) -> bool:
    """_summary_

    Args:
        hands (list): _description_
        hit_point (list, optional): _description_. Defaults to [0.5, 1].
        radius (int, optional): _description_. Defaults to 0.1.

    Returns:
        bool: _description_
    """
    if np.linalg.norm(np.array([hands[0], hands[1]]) - np.array(hit_point)) <= hit_radius:
        flg = True
    elif np.linalg.norm(np.array([hands[3], hands[4]]) - np.array(hit_point)) <= hit_radius:
        flg = True
    else:
        flg = False
    return flg

    
def detect_guard(nose: list, hands: list) -> bool:
    if nose[1] >= hands[1] and nose[1] >= hands[4] and\
            abs(hands[0] - hands[3]) <= 0.4 and hands[1] <= 0.6 and hands[4] <= 0.6:
        flg = True
    else:
        flg = False
    return flg


def detect_punch(hands: list, lshoulder, rshoulder, lelbow, relbow) -> bool:
    if (hands[1] <= 0.4 or hands[4] <= 0.4) and\
        (np.linalg.norm(np.array(lshoulder) - np.array(lelbow)) <= 0.2 or
         np.linalg.norm(np.array(rshoulder) - np.array(relbow)) <= 0.2):
        flg = True
    else:
        flg = False
    return flg


class VideoProcessor:

    def __init__(self) -> None:
        self.result_queue = queue.Queue()

    def recv(self, frame):
        self.hit_threshould = 0.15
        self.hit_point: list = [0.5, 0.8]  # width, height
        self.hit_radius = 50
        img = frame.to_ndarray(format='bgr24')
        annotated_img = img.copy()
        annotated_img = cv2.flip(annotated_img, 1)
        results = detect_pose(annotated_img)
        if results.pose_landmarks is not None:
            hands = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].z,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].z,]
            nose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,]
            lshoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y, ]
            rshoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, ]
            lelbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y, ]
            relbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y, ]
            cv2.circle(annotated_img,
                       center=(int(self.hit_point[0]*annotated_img.shape[1]),
                               int(self.hit_point[1]*annotated_img.shape[0]),),
                       radius=self.hit_radius,
                       color=(0, 255, 0),
                       thickness=3,
                       lineType=cv2.LINE_4)
            annotated_img = plot_to_img(annotated_img, results)
            if detect_hit(hands, self.hit_point, self.hit_radius/annotated_img.shape[0]):
                cv2.putText(annotated_img, 'Hit!', (
                    int(self.hit_point[0]*annotated_img.shape[1]), int(self.hit_point[1]*annotated_img.shape[0])),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5, cv2.LINE_AA)
            if detect_guard(nose, hands):
                cv2.putText(annotated_img, 'Guard', (
                    int(0.5*annotated_img.shape[0]), int(0.5*annotated_img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 255), 5, cv2.LINE_AA)
            if detect_punch(hands, lshoulder, rshoulder, lelbow, relbow):
                cv2.putText(annotated_img, 'Punch', (
                    int(0.5*annotated_img.shape[0]), int(0.5*annotated_img.shape[1])),
                    cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 255), 5, cv2.LINE_AA)
            self.result_queue.put(hands)
        img_dst = annotated_img
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
