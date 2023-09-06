import cv2
import time

# 영상 검출기
def video_detector(cascade, video_file):
    # 영상 파일 열기
    cap = cv2.VideoCapture(video_file)
    
    while True:
        start_time = time.time()
        
        # 영상 프레임 읽기
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 그레이 스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)
        
        # FPS 계산
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 영상 출력
        cv2.imshow('Face Detection', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 영상 파일 닫기
    cap.release()
    cv2.destroyAllWindows()

# 이미지 검출기
def img_detector(cascade, image_file):
    # 이미지 파일 읽기
    img = cv2.imread(image_file)
    
    # 그레이 스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)
    
    # 이미지 출력
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 가중치 파일 경로
    cascade_filename = 'haarcascade_frontalface_alt.xml'
    
    # 모델 불러오기
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_filename)

    # 영상 파일 경로 (영상 파일이름을 'sample.mp4'로 가정합니다.)
    video_file = 'sample.mp4'
    
    # 이미지 파일 경로 (이미지 파일이름을 'sample.jpg'로 가정합니다.)
    image_file = 'sample.jpg'
    
    # 영상 탐지기 실행
    video_detector(cascade, video_file)
    
    # 이미지 탐지기 실행
    img_detector(cascade, image_file)
