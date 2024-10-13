import cv2

face_ref = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=5)
    return faces

def drawer_block(frame):
    for x, y, w, h in face_detection(frame):
        # Menambahkan kotak frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 0, 0), 2)
        # menambahkan nama
        cv2.putText(frame, "Muhammad Fikri", (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, (0, 0, 0), 2)
    return frame

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        frame = drawer_block(frame)
        cv2.imshow("Pyface", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()
            break

if __name__ == '__main__':
    main()