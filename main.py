"""Camera face recognition. Importing this module never starts the camera."""

import argparse
import math
from numbers import Real
import os
from pathlib import Path
import pickle
import tempfile


PROJECT_DIR = Path(__file__).resolve().parent
DESCRIPTOR_SIZE = 128
DEFAULT_TOLERANCE = 0.6


def validate_embedding(values):
    """Accept finite, 128-dimensional descriptors, including legacy NumPy vectors."""
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("Ожидается вектор лица из 128 чисел") from error
    if len(values) != DESCRIPTOR_SIZE:
        raise ValueError("Ожидается вектор лица из 128 чисел")
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(value) for value in values):
        raise ValueError("Вектор лица должен содержать только конечные числа")
    return tuple(float(value) for value in values)


def validate_known_faces(faces):
    if not isinstance(faces, dict):
        raise ValueError("База лиц должна содержать словарь с именами и векторами")
    result = {}
    for name, embedding in faces.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Имя лица не должно быть пустым")
        result[name] = validate_embedding(embedding)
    return result


def load_known_faces(path):
    """Read the legacy dictionary format. Only use a trusted local pickle file."""
    try:
        with Path(path).open("rb") as source:
            return validate_known_faces(pickle.load(source))
    except FileNotFoundError:
        return {}


def save_known_faces(path, faces):
    """Replace the database only after the complete new contents are written."""
    faces = validate_known_faces(faces)
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            pickle.dump(faces, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def is_match(embedding1, embedding2, tolerance=DEFAULT_TOLERANCE):
    return math.dist(embedding1, embedding2) < tolerance


def identify_face(embedding, known_faces, tolerance=DEFAULT_TOLERANCE):
    """Choose the nearest match, independently of database insertion order."""
    best_name = "Неизвестно"
    best_distance = tolerance
    for name, candidate in known_faces.items():
        candidate_distance = math.dist(embedding, candidate)
        if candidate_distance < best_distance:
            best_name, best_distance = name, candidate_distance
    return best_name


def get_face_embedding(rgb_frame, face, predictor, face_recognizer):
    """Use the specific detected rectangle and the unannotated RGB frame."""
    landmarks = predictor(rgb_frame, face)
    return validate_embedding(face_recognizer.compute_face_descriptor(rgb_frame, landmarks))


def enroll_face(embeddings, known_faces, database_path, input_fn=input, output_fn=print):
    """Enrollment is permitted only for one face from the current frame."""
    if len(embeddings) != 1:
        output_fn("Для сохранения в текущем кадре должно быть ровно одно лицо.")
        return False
    name = input_fn("Введите имя для сохранения: ").strip()
    if not name:
        output_fn("Пустое имя: лицо не сохранено.")
        return False
    if name in known_faces:
        answer = input_fn(f"Имя {name} уже существует. Заменить? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "д", "да"}:
            output_fn("Сохранение отменено.")
            return False
    updated_faces = dict(known_faces)
    updated_faces[name] = validate_embedding(embeddings[0])
    save_known_faces(database_path, updated_faces)
    known_faces.clear()
    known_faces.update(updated_faces)
    output_fn(f"Лицо {name} сохранено!")
    return True


def run_camera(cv2, detector, predictor, face_recognizer, known_faces, database_path,
               camera_index=0, tolerance=DEFAULT_TOLERANCE, input_fn=input, output_fn=print):
    """Keep capture/model dependencies injectable for offline synthetic tests."""
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            output_fn("Не удалось открыть камеру")
            return 1
        output_fn("Нажмите 's', чтобы сохранить одно лицо в кадре, 'q' для выхода.")
        while True:
            success, frame = cap.read()
            if not success:
                output_fn("Не удалось получить кадр с камеры")
                return 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = list(detector(rgb_frame))
            embeddings = [get_face_embedding(rgb_frame, face, predictor, face_recognizer)
                          for face in faces]
            # Extract every descriptor before adding annotations to the display frame.
            for face, embedding in zip(faces, embeddings):
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                name_label = identify_face(embedding, known_faces, tolerance)
                cv2.putText(frame, name_label, (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s"):
                try:
                    enroll_face(embeddings, known_faces, database_path, input_fn, output_fn)
                except (OSError, ValueError) as error:
                    output_fn(f"Не удалось сохранить лицо: {error}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Распознавание лиц с камеры OpenCV/dlib")
    parser.add_argument("--camera", type=int, default=0, help="Индекс камеры (по умолчанию 0)")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_DIR,
                        help="Каталог с двумя моделями dlib")
    parser.add_argument("--database", type=Path, default=PROJECT_DIR / "known_faces.pkl",
                        help="Доверенный локальный файл pickle для регистрации лиц")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="Максимальное расстояние совпадения (по умолчанию 0.6)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance должен быть положительным конечным числом")
    if args.camera < 0:
        parser.error("--camera должен быть неотрицательным")

    model_paths = [args.models_dir / name for name in (
        "shape_predictor_68_face_landmarks.dat", "dlib_face_recognition_resnet_model_v1.dat",
    )]
    try:
        for path in model_paths:
            if not path.is_file():
                raise OSError(f"Не найдена модель: {path}")
        known_faces = load_known_faces(args.database)
        import cv2
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(model_paths[0]))
        face_recognizer = dlib.face_recognition_model_v1(str(model_paths[1]))
        return run_camera(cv2, detector, predictor, face_recognizer, known_faces,
                          args.database, args.camera, args.tolerance)
    except (ImportError, OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        print(f"Ошибка запуска или обработки: {error}")
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
