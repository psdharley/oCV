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
import dlib
            # 003 : 여기선 얼굴인식 용으로 사용한다
            # 이미지 처리, 선형대수 뿐만 아니라 다양한 머신러닝 알고리즘을 활용할 수 있는 dlib 라이브러리는 C++로 작성된 툴킷이지만,
            # python 패키지로도 설치해 사용할 수 있다.
            # 특히 HOG(Histogram of Oriented Gradients) 특성을 사용하여 얼굴 검출하는 기능이 많이 사용되고 소개된다.
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

import numpy as np  # 004
import sys          # 006



# 001
# load video : 동영상을 가져오거나 웹캠으로 실시간 화면보기
# 파일 이름대신 0을 넣으면 웹캠이 켜지고 자신의 얼굴로 테스트 가능
cap = cv2.VideoCapture('samples/girl.mp4')

# 002
# 스케일로 크기 조절 상수
scaler = 0.3    # 10 분의 3으로 줄이기 위해

# 003
# initialize 'face detector' and 'shape predictor' : 얼굴을 찾아주는 모듈 초기화
detector = dlib.get_frontal_face_detector()     # 얼굴 디텍터 모듈을 디텍터라는 변수이름으로 초기화
# 이미 머신러닝으로 학습된 모델인 shape_predictor 가져와 사용하기 위해 초기화
# .dat 는 머신러닝으로 학습된 모델 파일이다. 이것을 shape_predictor() 파라미터로 넣어 초기화
# 이모델로 얼굴인식이 가능함
predictor = dlib.shape_predictor(r'D:\dev\wsPy\project\oCV\001_app\shape_predictor\shape_predictor_68_face_landmarks.dat')
# D:\dev\wsPy\project\oCV\001_app\shape_predictor 에 백업해 둠

# 006
# load overlay image
# cv2.IMREAD_UNCHANGED 파일 이미지를 BGRA 타입으로 읽기 (알파채널까지 읽기)
overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)



# 006
# overlay function
# 우선 오버레이 할 이미지는 투명한 배경이미지로 꼭 PNG 를 사용해야 함
# 이미지를 동영상에 띄우는 함수는 어렵다 (구글링에서 찾은것을 사용한다)
# 함수는 라이언 이미지를 센터x 센터y 를 중심으로 놓고,
# 오버레이 사이즈만큼 리사이즈해서 원본 이미지에다가 넣어준다(얼굴크기만큼 리사이즈 해야한다)
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



# 001
# loop
while True:  # 프래임 단위로 계속 읽어야 하므로 구간반복..

    # 001
    # read frame buffer from video
    ret, img = cap.read()  # 동영상파일에서 프래임 단위로 읽기
    if not ret:  # 프래임이 없으면 break로 프로그램 종료시키기
        break

    # 002
    # resize frame
    # 화면에 보여주기 전에 동영상의 가로세로 크기를 줄인다.
    # 위쪽 상수 scaler 스케일러 상수가 사용된다. 10분의 3으로 줄이기
    # cv2.resize(img, dsize) 명령어는 -> img 를 dsize 크기로 조절 (resize)
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

    # 003
    # 위의 가로세로가 줄여진 이미지를 ori 라는 변수에 복사해놓기
    ori = img.copy()

    # 003
    # detect faces
    # 디텍터에 이미지만 넣어 주기만 하면 얼굴이 인식된다.
    faces = detector(img)
    # 여러 얼굴이 동시에 나오기 때문에 face 라는 변수를 지정해서 얼굴 하나만 0번 인덱스만 저장하기로 한다
    face = faces[0]

    # 003
    # 네모를 얼굴에 맞춰 보이게 하기
    img = cv2.rectangle(img,
                        pt1=(face.left(), face.top()),      # 좌상단
                        pt2=(face.right(), face.bottom()),  # 우하단
                        color=(255,255,255),                # 네모의 섹상 하얀색
                        thickness=2,                        # 선의 두께
                        lineType=cv2.LINE_AA                # 선의 타입
                        )

    # 004
    # 얼굴의 특징점을 찾기위해 프레딕터를 사용한다.
    # 프레딕터에는 이미지와 구한 얼굴영역이 들어간다.
    dlib_shape = predictor(img, face)
    # dlib 섀이프를 리턴받게 되있는, 연산을 쉽게하기위해 넘파이 어레이로 바꾸어서 shape_2d 에 저장한다.
    shape_2d = np.array([ [p.x, p.y] for p in dlib_shape.parts() ])

    # 004
    # 얼굴 얼굴외과, 눈썹, 눈, 코, 입등 에 점(서클)로 표시해서 보이게 하기
    # 얼굴 특징점의 갯수는 68개 인데, 위의 while 이 한번 돌때 점68개는
    # for 루프를 돌면서 68개의 점을 openCV 에 cirecle 이라는 메소드를 사용하여 그리도록 함
    for s in shape_2d:
        cv2.circle(img,
                   center=tuple(s),         # 원 중앙값
                   radius=1,                # 반지름
                   color=(255,255,255),     # 원의 색상
                   thickness=2,             # 선 두께
                   lineType=cv2.LINE_AA     # 선 스타일
                   )

    # 005
    # compute center of face
    # 얼굴의 좌상단 우하단의 점을 구하기
    top_left = np.min(shape_2d, axis=0)
    bot_righ = np.max(shape_2d, axis=0)
    # 얼굴의 중심구하기
    # 모든 특징점의 평균을(np.mean) 구해서 얼굴의 중심구하기
    # 센터x 와 센터y를 구한다음에 소수점일수도 있으니까 정수형 int 타입으로 변환
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int64)     # int로만 하면 경고뜸 int64로 해야한다

    # 005
    # 얼굴의 좌상단 우하단의 점을 구한것을, 이미지에 표시해서 보이게 하기
    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bot_righ), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # 006
    # 오버레이 사이즈만큼 리사이즈해서 원본 이미지에다가 넣어 주기 위해 (라이언을 얼굴크기만큼 리사이즈 해야한다)
    # 우하단에서 - 좌상단 좌표를 뺀 (x,y) 길이의 가장 긴 값 구해서 변수에 저장하기
    #### face_size = max(bot_righ - top_left)
    # 위의 소스로 인해 혹 이미지가 조금 작을 경우라 생각 된다면 아래와 같이 1.8 정도를 곱해주고, 소수점이 나올수도 있으니 정수 int 로 변환하자.
    # 위쪽 사이즈와 현재 아래부분 사이즈 둘 중 하나만 사용하자
    face_size = int(max(bot_righ - top_left) * 1.8)

    # 구글링한 함수에 넣어준다 (이미지를 동영상에 띄우는 함수)
    ### result = overlay_transparent(ori, overlay, center_x, center_y, overlay_size=(face_size, face_size))
    # 라이언의 얼굴이 위치를 변경하려면 아래와 같이 함수 보내는 값을 달리하면 된다.
    # center_x + 25 ---> 오른쪽 우측으로 25 정도 이동
    # center_y - 25 ---> 윗 쪽으로 25 정도 이동
    result = overlay_transparent(ori, overlay, center_x+25, center_y-25, overlay_size=(face_size, face_size))

    # 001
    # visualize : 윈도우를 띄우고 동영상을 보여주기
    # cv2.imshow('윈도우이름', 이미지)
    cv2.imshow('original', img)     # 원본
    # 006
    cv2.imshow('result', result)    # 라이언 오버레이



    # 001
    # cv2.waitKey(1)    아래 소스로 변경하기
    # 006
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)
