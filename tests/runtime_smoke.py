"""Opt-in runtime checks using generated pixels, never a camera or stored faces.

Run explicitly with ``python tests/runtime_smoke.py -v`` after installing the
requirements. Standard-library-only unit-test discovery does not run this file.
"""

from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import dlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class RuntimeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = dlib.get_frontal_face_detector()
        cls.predictor = dlib.shape_predictor(
            str(main.PROJECT_DIR / "shape_predictor_68_face_landmarks.dat")
        )
        cls.recognizer = dlib.face_recognition_model_v1(
            str(main.PROJECT_DIR / "dlib_face_recognition_resnet_model_v1.dat")
        )

    def generated_frame(self):
        # Deliberately artificial pixels; no images, database, or capture device.
        return np.random.default_rng(2026).integers(0, 256, (256, 256, 3), dtype=np.uint8)

    def test_detector_accepts_opencv_rgb_array_without_a_face(self):
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
        self.assertEqual(list(self.detector(rgb)), [])

    def test_predictor_and_recognizer_use_each_supplied_rectangle(self):
        rgb = cv2.cvtColor(self.generated_frame(), cv2.COLOR_BGR2RGB)
        rectangles = [dlib.rectangle(20, 30, 115, 150), dlib.rectangle(140, 50, 235, 170)]
        embeddings = []
        for rectangle in rectangles:
            with self.subTest(rectangle=rectangle):
                landmarks = self.predictor(rgb, rectangle)
                self.assertEqual(landmarks.num_parts, 68)
                expected = tuple(self.recognizer.compute_face_descriptor(rgb, landmarks))
                actual = main.get_face_embedding(rgb, rectangle, self.predictor, self.recognizer)
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), 128)
                self.assertTrue(np.isfinite(actual).all())
                embeddings.append(actual)
        self.assertGreater(np.linalg.norm(np.subtract(*embeddings)), 0.01)

    def test_real_opencv_operations_and_models_in_camera_loop_without_capture(self):
        frame = self.generated_frame()
        original = frame.copy()
        rectangle = dlib.rectangle(20, 30, 115, 150)
        expected = main.get_face_embedding(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB), rectangle, self.predictor, self.recognizer
        )

        class GeneratedCapture:
            released = False
            last_frame = None

            def isOpened(self):
                return True

            def read(self):
                self.last_frame = original.copy()
                return True, self.last_frame

            def release(self):
                self.released = True

        capture = GeneratedCapture()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "synthetic.pkl"
            with patch.object(cv2, "VideoCapture", return_value=capture) as open_capture, \
                    patch.object(cv2, "imshow") as show, \
                    patch.object(cv2, "waitKey", side_effect=[ord("s"), ord("q")]), \
                    patch.object(cv2, "destroyAllWindows") as close_windows:
                status = main.run_camera(
                    cv2, lambda rgb: [rectangle], self.predictor, self.recognizer,
                    {}, database, input_fn=lambda prompt: "Generated fixture",
                    output_fn=lambda message: None,
                )
            self.assertEqual(status, 0)
            self.assertEqual(main.load_known_faces(database), {"Generated fixture": expected})
            open_capture.assert_called_once_with(0)
            self.assertEqual(show.call_count, 2)
            close_windows.assert_called_once_with()
        self.assertTrue(capture.released)
        self.assertFalse(np.array_equal(capture.last_frame, original))

    def test_legacy_numpy_vectors_round_trip_as_plain_tuples(self):
        for dtype in (np.float32, np.float64, np.int64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "synthetic.pkl"
                values = np.arange(128, dtype=dtype)
                with database.open("wb") as destination:
                    pickle.dump({"Generated fixture": values}, destination)
                loaded = main.load_known_faces(database)
                self.assertEqual(loaded, {"Generated fixture": tuple(map(float, values))})
                main.save_known_faces(database, loaded)
                with database.open("rb") as source:
                    saved = pickle.load(source)
                self.assertIs(type(saved["Generated fixture"]), tuple)
                self.assertEqual(saved, loaded)


if __name__ == "__main__":
    unittest.main()
