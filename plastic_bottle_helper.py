import tensorflow # 딥러닝
import numpy as np # 수치계산
import cv2 # opencv2
import pyautogui  # 마우스 x, y 좌표 확인용
import time # 시간 사용

WINDOW_NAME = 'plasticBottleHelper'
check_screen = 1 # 1: 카메라 화면, 2: 라벨 X, 3: 라벨 O
prev_time = 0
FPS = 10
prediction_value = 0.98

# 모델 위치
model_filename = '/plasticBottleHelper/converted_keras/keras_model.h5'
img_filename1 = '/plasticBottleHelper/image1.png'
img_filename2 = '/plasticBottleHelper/image2.png'

# 케라스 모델 가져오기
model = tensorflow.keras.models.load_model(model_filename)

# 카메라를 제어할 수 있는 객체
# 외부 웹캠으로 비디오 캡처 초기화
capture = cv2.VideoCapture(1)
# 외부 웹캠이 없다면 내장 웹캠을 사용
if not capture.read()[0]:
    capture = cv2.VideoCapture(0)

# Full screen mode
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 마우스 이벤트
def mouse_event(event, x, y, flags, param):  
    global check_screen
     
    if event == cv2.EVENT_LBUTTONDOWN:
        if (check_screen != 1) & (1280 < X < 2515) & (1140 < Y < 1412):
            check_screen = 1

while True:
    # 프레임 계산 10fps
    current_time = time.time() - prev_time
    
    if check_screen == 1:
        # 비디오를 한 프레임씩 읽기
        ret, frame = capture.read()
        if not ret:
            break

        # 비디오 좌우 반전
        # frame = cv2.flip(frame, 1)
        # 비디오 상하 반전
        # frame = cv2.flip(frame, 0)

        # 비디오 크기 재설정
        frame_resize = frame[:, 80:80+frame.shape[0]]
        frame_input = cv2.resize(frame_resize, (224, 224))

        frame_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        frame_input = (frame_input.astype(np.float32) / 127.0) - 1
        frame_input = np.expand_dims(frame_input, axis=0)

        prediction = model.predict(frame_input)
        
        cv2.rectangle(frame, (80, 0), (80+frame.shape[0], frame.shape[0]), (0, 0, 255), 5)
        cv2.putText(frame, str(round(prediction[0, 0], 5)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        cv2.putText(frame, str(round(prediction[0, 1], 5)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        cv2.putText(frame, str(round(prediction[0, 2], 5)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        
    else:
        X, Y = pyautogui.position()
        H, W = frame.shape[:2]
        
        if check_screen == 2:
            frame = cv2.imread(img_filename1)
        else: # if check_screen == 3:
            frame = cv2.imread(img_filename2)
        
        # 버튼        
        cv2.setMouseCallback(WINDOW_NAME, mouse_event, frame)
        cv2.putText(frame, "X : " + str(X), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        cv2.putText(frame, "Y : " + str(Y), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        cv2.putText(frame, "H : " + str(H), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        cv2.putText(frame, "W : " + str(W), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

    # 종료 버튼 0xFF == 64bit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        check_screen = 1

    if(prediction[0, 0] > prediction[0, 1]):
        if (prediction[0, 0] > prediction_value):
            check_screen = 2
            prediction[0, 0] = 0
            prediction[0, 1] = 0

    if(prediction[0, 1] > prediction[0, 0]):
        if (prediction[0, 1] > prediction_value):
            check_screen = 3
            prediction[0, 0] = 0
            prediction[0, 1] = 0

    # 출력
    if (ret is True) & (current_time > 1./ FPS) :
        prev_time = time.time()
        cv2.imshow(WINDOW_NAME, frame)

# 비디오 캡처 개체 해제
capture.release()
cv2.destroyAllWindows()