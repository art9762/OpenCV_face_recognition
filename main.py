import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pickle

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

try:
    with open('known_faces.pkl', 'rb') as file:
        known_faces = pickle.load(file)
except FileNotFoundError:
    known_faces = {}


def get_face_embedding(face_image):
    face_landmarks = predictor(face_image, detector(face_image)[0])
    return np.array(face_recognizer.compute_face_descriptor(face_image, face_landmarks))


def is_match(embedding1, embedding2):
    return distance.euclidean(embedding1, embedding2) < 0.6


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

print("Нажмите 's', чтобы сохранить новое лицо, 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_embedding = get_face_embedding(frame)

        name_label = "Неизвестно"
        for name, embedding in known_faces.items():
            if is_match(face_embedding, embedding):
                name_label = name
                break

        cv2.putText(frame, name_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        name = input("Введите имя для сохранения: ")
        if name:
            known_faces[name] = face_embedding
            with open('known_faces.pkl', 'wb') as file:
                pickle.dump(known_faces, file)
            print(f"Лицо {name} сохранено!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()