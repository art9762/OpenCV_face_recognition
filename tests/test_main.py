import contextlib
import io
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main


def embedding(value=0.0):
    return (float(value),) + (0.0,) * 127


class Face:
    def __init__(self, value):
        self.value = value

    def left(self):
        return 10

    def top(self):
        return 20

    def width(self):
        return 30

    def height(self):
        return 40


class FakeCV2:
    COLOR_BGR2RGB = "BGR2RGB"
    FONT_HERSHEY_SIMPLEX = "font"

    def __init__(self, faces_per_frame, keys, opened=True):
        self.frames = iter(SimpleNamespace(faces=faces, annotated=False, color="bgr")
                           for faces in faces_per_frame)
        self.keys = iter(keys)
        self.opened = opened
        self.released = False
        self.destroyed = False
        self.labels = []
        self.events = []
        self.camera_index = None

    def VideoCapture(self, index):
        self.camera_index = index
        return self

    def isOpened(self):
        return self.opened

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def cvtColor(self, frame, conversion):
        if frame.color != "bgr" or conversion != self.COLOR_BGR2RGB:
            raise AssertionError("The model must receive RGB converted from the camera BGR frame")
        return SimpleNamespace(faces=frame.faces, source=frame, color="rgb")

    def rectangle(self, frame, *args):
        frame.annotated = True
        self.events.append("draw")

    def putText(self, frame, label, *args):
        self.labels.append(label)

    def imshow(self, title, frame):
        pass

    def waitKey(self, delay):
        return next(self.keys)

    def release(self):
        self.released = True

    def destroyAllWindows(self):
        self.destroyed = True


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "synthetic.pkl"
        self.output = Mock()

    def run_fake(self, cv2, known_faces=None, input_fn=None, predictor=None):
        def detect(frame):
            self.assertEqual(frame.color, "rgb")
            return frame.faces

        def landmarks(frame, face):
            self.assertEqual(frame.color, "rgb")
            self.assertFalse(frame.source.annotated)
            cv2.events.append(f"landmarks:{face.value}")
            return face

        def descriptor(frame, face):
            self.assertFalse(frame.source.annotated)
            cv2.events.append(f"descriptor:{face.value}")
            return embedding(face.value)

        self.detector = Mock(side_effect=detect)
        recognizer = SimpleNamespace(compute_face_descriptor=descriptor)
        return main.run_camera(
            cv2, self.detector, predictor or landmarks, recognizer,
            {} if known_faces is None else known_faces, self.database,
            camera_index=7, input_fn=input_fn or Mock(side_effect=AssertionError("Unexpected input")),
            output_fn=self.output,
        )

    def test_each_face_uses_its_rectangle_and_rgb_before_annotations(self):
        cv2 = FakeCV2([[Face(0), Face(2)]], [ord("q")])
        self.assertEqual(self.run_fake(cv2, {"Alice": embedding(0), "Bob": embedding(2)}), 0)
        self.assertEqual(cv2.labels, ["Alice", "Bob"])
        self.detector.assert_called_once()
        self.assertEqual(cv2.events, ["landmarks:0", "descriptor:0", "landmarks:2",
                                      "descriptor:2", "draw", "draw"])
        self.assertEqual(cv2.camera_index, 7)
        self.assertTrue(cv2.released and cv2.destroyed)

    def test_save_without_any_face_does_not_prompt_or_create_database(self):
        cv2 = FakeCV2([[], []], [ord("s"), ord("q")])
        self.assertEqual(self.run_fake(cv2), 0)
        self.assertFalse(self.database.exists())
        self.assertIn("ровно одно лицо", self.output.call_args_list[1].args[0])

    def test_save_does_not_reuse_previous_frames_embedding(self):
        cv2 = FakeCV2([[Face(0)], [], []], [-1, ord("s"), ord("q")])
        self.assertEqual(self.run_fake(cv2), 0)
        self.assertFalse(self.database.exists())

    def test_multiple_faces_block_registration(self):
        cv2 = FakeCV2([[Face(0), Face(2)], []], [ord("s"), ord("q")])
        self.assertEqual(self.run_fake(cv2), 0)
        self.assertFalse(self.database.exists())

    def test_single_face_registration_uses_the_current_descriptor(self):
        cv2 = FakeCV2([[Face(3)], []], [ord("s"), ord("q")])
        known_faces = {}
        self.assertEqual(self.run_fake(cv2, known_faces, Mock(return_value=" Test ")), 0)
        self.assertEqual(known_faces, {"Test": embedding(3)})
        self.assertEqual(main.load_known_faces(self.database), known_faces)

    def test_failed_registration_is_reported_and_capture_can_exit(self):
        cv2 = FakeCV2([[Face(3)], []], [ord("s"), ord("q")])
        known_faces = {}
        with patch.object(main, "save_known_faces", side_effect=OSError("disk full")):
            self.assertEqual(self.run_fake(cv2, known_faces, Mock(return_value="Test")), 0)
        self.assertEqual(known_faces, {})
        self.assertIn("disk full", self.output.call_args_list[-1].args[0])
        self.assertTrue(cv2.released and cv2.destroyed)

    def test_unopened_camera_is_released(self):
        cv2 = FakeCV2([], [], opened=False)
        self.assertEqual(self.run_fake(cv2), 1)
        self.assertTrue(cv2.released and cv2.destroyed)

    def test_failed_frame_read_is_reported_and_released(self):
        cv2 = FakeCV2([], [])
        self.assertEqual(self.run_fake(cv2), 1)
        self.assertTrue(cv2.released and cv2.destroyed)

    def test_model_exception_still_releases_camera(self):
        cv2 = FakeCV2([[Face(0)]], [])
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            self.run_fake(cv2, predictor=Mock(side_effect=RuntimeError("model failure")))
        self.assertTrue(cv2.released and cv2.destroyed)

    def test_interrupted_enrollment_still_releases_camera(self):
        cv2 = FakeCV2([[Face(0)]], [ord("s")])
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(cv2, input_fn=Mock(side_effect=KeyboardInterrupt))
        self.assertTrue(cv2.released and cv2.destroyed)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "synthetic.pkl"

    def test_missing_database_is_empty(self):
        self.assertEqual(main.load_known_faces(self.database), {})

    def test_legacy_dict_of_iterable_vectors_can_be_read(self):
        with self.database.open("wb") as destination:
            pickle.dump({"Test": list(embedding(2))}, destination)
        self.assertEqual(main.load_known_faces(self.database), {"Test": embedding(2)})

    def test_database_round_trip_preserves_pickle_dictionary_format(self):
        main.save_known_faces(self.database, {"Тест": embedding(3)})
        self.assertEqual(main.load_known_faces(self.database), {"Тест": embedding(3)})
        with self.database.open("rb") as source:
            self.assertEqual(pickle.load(source), {"Тест": embedding(3)})

    def test_failed_replace_preserves_database_and_removes_temporary_file(self):
        main.save_known_faces(self.database, {"Original": embedding(0)})
        original = self.database.read_bytes()
        with patch.object(main.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                main.save_known_faces(self.database, {"New": embedding(1)})
        self.assertEqual(self.database.read_bytes(), original)
        self.assertEqual(list(self.database.parent.iterdir()), [self.database])

    def test_failed_enrollment_keeps_in_memory_identity_unchanged(self):
        known_faces = {"Test": embedding(0)}
        with patch.object(main, "save_known_faces", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                main.enroll_face([embedding(1)], known_faces, self.database,
                                 Mock(side_effect=["Test", "yes"]), Mock())
        self.assertEqual(known_faces, {"Test": embedding(0)})

    def test_existing_name_is_not_overwritten_without_confirmation(self):
        known_faces = {"Test": embedding(0)}
        self.assertFalse(main.enroll_face([embedding(1)], known_faces, self.database,
                                         Mock(side_effect=["Test", ""]), Mock()))
        self.assertEqual(known_faces, {"Test": embedding(0)})
        self.assertFalse(self.database.exists())

    def test_existing_name_can_be_replaced_after_confirmation(self):
        known_faces = {"Test": embedding(0)}
        self.assertTrue(main.enroll_face([embedding(1)], known_faces, self.database,
                                        Mock(side_effect=["Test", "да"]), Mock()))
        self.assertEqual(main.load_known_faces(self.database), {"Test": embedding(1)})

    def test_empty_name_cannot_be_saved(self):
        self.assertFalse(main.enroll_face([embedding(0)], {}, self.database,
                                         Mock(return_value="  "), Mock()))
        self.assertFalse(self.database.exists())

    def test_invalid_database_shapes_and_nonfinite_values_are_rejected(self):
        for invalid in [[], {"": embedding()}, {"Test": [0.0]},
                        {"Test": [float("nan")] * 128}, {"Test": [float("inf")] * 128},
                        {"Test": [True] * 128}, {"Test": ["0"] * 128}, {"Test": None}]:
            with self.subTest(invalid_type=type(invalid)):
                with self.database.open("wb") as destination:
                    pickle.dump(invalid, destination)
                with self.assertRaises(ValueError):
                    main.load_known_faces(self.database)


class MatchingAndEntryPointTests(unittest.TestCase):
    def test_nearest_match_wins_instead_of_first_acceptable_entry(self):
        self.assertEqual(main.identify_face(embedding(), {"Farther": embedding(0.5),
                                                         "Nearer": embedding(0.1)}), "Nearer")

    def test_threshold_is_strict_and_unknown_is_reported(self):
        self.assertFalse(main.is_match(embedding(), embedding(0.6)))
        self.assertEqual(main.identify_face(embedding(), {"Test": embedding(0.6)}), "Неизвестно")
        self.assertEqual(main.identify_face(embedding(), {}), "Неизвестно")

    def test_custom_tolerance_is_used(self):
        self.assertEqual(main.identify_face(embedding(), {"Test": embedding(0.2)}, 0.1), "Неизвестно")
        self.assertTrue(main.is_match(embedding(), embedding(0.2), 0.3))

    def test_import_and_help_do_not_import_runtime_dependencies(self):
        # Isolated process, no model, database, or camera calls; no third-party modules needed.
        result = subprocess.run([
            sys.executable, "-c",
            "import main, sys; assert 'cv2' not in sys.modules; "
            "assert 'dlib' not in sys.modules; main.main(['--help'])",
        ], cwd=main.PROJECT_DIR, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--database", result.stdout)

    def test_invalid_cli_options_stop_before_reading_database_or_models(self):
        for arguments in [["--camera", "-1"], ["--tolerance", "0"], ["--tolerance", "nan"],
                          ["--tolerance", "inf"]]:
            with self.subTest(arguments=arguments), patch.object(main, "load_known_faces") as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main.main(arguments)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()

    def test_missing_models_are_reported_before_reading_database(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "load_known_faces") as load, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(["--models-dir", directory]), 1)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
