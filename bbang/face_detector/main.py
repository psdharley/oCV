"""
얼굴인식 스노우 카메라 쉽게 따라만들기 - Python
https://www.youtube.com/watch?v=tpWVyJqehG4

- 전체 소스코드
https://github.com/kairess/face_detector

- 무료 동영상 다운로드
https://videos.pexels.com/search/face

- shape_predictor_68_face_landmarks.dat 다운로드
https://github.com/davisking/dlib-mod...
"""

import cv2  # openCV 3+ 설치하면된다
import dlib
            # 설치오류 이슈가 많다. cmake 먼저 설치 -> 그래도 안됨, 참고 https://updaun.tistory.com/entry/python-python-37-dlib-install-error
            # D:\dev\wsPy\project\oCV\dlib설치오류\ 에 있는 dlib-19.17.0-cp37-cp37m-win_amd64.whl 아래의 경로
            # D:\dev\wsPy\project\oCV\venv\Scripts\dlib-19.17.0-cp37-cp37m-win_amd64.whl 에 넣어두고
            # 파이참 터미널에서
            # D:\dev\wsPy\project\oCV\venv\Scripts> pip install dlib-19.17.0-cp37-cp37m-win_amd64.whl 실행
            # """ 설치 성공 메세지 아래 :
            # Processing d:\dev\wspy\project\ocv\venv\scripts\dlib-19.17.0-cp37-cp37m-win_amd64.whl
            # Installing collected packages: dlib
            # Successfully installed dlib-19.17.0
            # """
import numpy as np
import sys


# 스케일로 크기 조절 상수
scaler = 0.3    # 10 분의 3으로 줄이기 위해

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(r'D:\dev\wsPy\project\oCV\001_app\shape_predictor\shape_predictor_68_face_landmarks.dat')

# load video
# 파일 이름대신 0을 넣으면 웹캠이 켜지고 자신의 얼굴로 테스트 가능
cap = cv2.VideoCapture('samples/girl.mp4')

# load overlay image
overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)


# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


face_roi = []
face_sizes = []

# loop
while True:  # 프래임 단위로 계속 읽어야 하므로..
    # read frame buffer from video
    ret, img = cap.read()  # 동영상파일에서 프래임 단위로 읽기
    if not ret:  # 프래임이 없으면 break로 프로그램 종료시키기
        break

    # resize frame
    # 화면에 보여주기 전에 동영상의 가로세로 크기를 줄인다.
    # 위쪽 상수 scaler 스케일러 상수가 사용된다. 10분의 3으로 줄이기
    # cv2.resize(img, dsize) 명령어는 -> img 를 dsize 크기로 조절 (resize)
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    # find faces
    if len(face_roi) == 0:
        faces = detector(img, 1)
    else:
        roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        # cv2.imshow('roi', roi_img)
        faces = detector(roi_img)

    # no faces
    if len(faces) == 0:
        print('no faces!')

    # find facial landmarks
    for face in faces:
        if len(face_roi) == 0:
            dlib_shape = predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        else:
            dlib_shape = predictor(roi_img, face)
            shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # compute face center
        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        # compute face boundaries
        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)

        # draw min, max coords
        cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        # compute face size
        face_size = max(max_coords - min_coords)
        face_sizes.append(face_size)
        if len(face_sizes) > 10:
            del face_sizes[0]
        mean_face_size = int(np.mean(face_sizes) * 1.8)

        # compute face roi
        face_roi = np.array(
            [int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2),
             int(max_coords[0] + face_size / 2)])
        face_roi = np.clip(face_roi, 0, 10000)

        # draw overlay on face
        result = overlay_transparent(ori, overlay, center_x + 8, center_y - 25,
                                     overlay_size=(mean_face_size, mean_face_size))

    # visualize
    # cv2.imshow('윈도우이름', 이미지)
    cv2.imshow('original', ori)
    cv2.imshow('facial landmarks', img)
    cv2.imshow('result', result)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)
