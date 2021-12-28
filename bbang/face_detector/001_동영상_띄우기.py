"""
얼굴인식 스노우 카메라 쉽게 따라만들기 - Python
https://www.youtube.com/watch?v=tpWVyJqehG4

- 전체 소스코드
https://github.com/kairess/face_detector

- 무료 동영상 다운로드
https://videos.pexels.com/search/face

- shape_predictor_68_face_landmarks.dat 다운로드
https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
"""


import cv2
            # 001
            # openCV 3+ 설치하면된다
            # OpenCV(Open Source Computer Vision)은 실시간 컴퓨터 비전을 목적으로 한 프로그래밍 라이브러리이다.
            # 원래는 인텔이 개발하였다. 실시간 이미지 프로세싱에 중점을 둔 라이브러리이다.



# 001
# load video : 동영상을 가져오거나 웹캠으로 실시간 화면보기
# 파일 이름대신 0을 넣으면 웹캠이 켜지고 자신의 얼굴로 테스트 가능
cap = cv2.VideoCapture('samples/girl.mp4')



# 001
# loop
while True:  # 프래임 단위로 계속 읽어야 하므로 구간반복..

    # 001
    # read frame buffer from video
    ret, img = cap.read()  # 동영상파일에서 프래임 단위로 읽기
    if not ret:  # 프래임이 없으면 break로 프로그램 종료시키기
        break

    # 001
    # visualize : 윈도우를 띄우고 동영상을 보여주기
    # cv2.imshow('윈도우이름', 이미지)
    cv2.imshow('original', img)
    cv2.waitKey(1)
