import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
	{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title('WebCam App')
webrtc_streamer(key="sample",
	rtc_configuration=RTC_CONFIGURATION)