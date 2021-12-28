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

    # 001
    # visualize : 윈도우를 띄우고 동영상을 보여주기
    # cv2.imshow('윈도우이름', 이미지)
    cv2.imshow('original', img)
    cv2.waitKey(1)
