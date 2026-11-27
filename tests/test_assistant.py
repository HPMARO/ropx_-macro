import ctypes
import queue
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from slime_assistant.vision import Template, Detector, Detection, Tracker
from slime_assistant.engine import Engine, Options
from slime_assistant import windows as win


class VisionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.image = rng.integers(30, 230, (40, 50, 3), dtype=np.uint8)
        cv2.ellipse(self.image, (25, 20), (20, 15), 0, 0, 360, (30, 190, 50), 2)
        self.detector = Detector([Template('slime', self.image)])

    def test_scaled_target_coordinates_and_roi(self):
        for scale in (.8, 1., 1.2):
            frame = np.zeros((400, 600, 3), np.uint8)
            target = cv2.resize(self.image, None, fx=scale, fy=scale)
            h, w = target.shape[:2]
            frame[160:160+h, 230:230+w] = target
            matches = self.detector.detect(frame, (.1, .1, .9, .9), .84)
            self.assertTrue(matches, scale)
            self.assertLessEqual(abs(matches[0].x - 230), 1)
            self.assertLessEqual(abs(matches[0].y - 160), 1)
            self.assertEqual([], self.detector.detect(frame, (0., 0., .2, .2), .84))

    def test_background_rejected(self):
        frame = np.random.default_rng(22).integers(0, 256, (400, 600, 3), dtype=np.uint8)
        self.assertEqual([], self.detector.detect(frame, (0., 0., 1., 1.), .84))

    def test_blank_template_rejected(self):
        with self.assertRaises(ValueError):
            Template('blank', np.zeros((40, 40, 3), np.uint8))

    def test_track_loss_requires_new_confirmations(self):
        tracker = Tracker()
        a = Detection(240, 160, 50, 40, .99, 'slime')
        for expected in (1, 2, 3):
            self.assertEqual(expected, tracker.update([a], (400, 600, 3))[1])
        self.assertEqual((None, 0), tracker.update([], (400, 600, 3)))
        self.assertEqual(1, tracker.update([a], (400, 600, 3))[1])

    def test_current_center_target_wins_over_previous_side_target(self):
        tracker = Tracker()
        side = Detection(450, 160, 50, 40, .99, 'slime')
        center = Detection(275, 180, 50, 40, .98, 'slime')
        tracker.update([side], (400, 600, 3))
        selected, hits = tracker.update([side, center], (400, 600, 3))
        self.assertIs(center, selected)
        self.assertEqual(1, hits)

    def test_switching_visible_slime_restarts_confirmation(self):
        tracker = Tracker()
        a = Detection(275, 180, 50, 40, .99, 'blue')
        b = Detection(275, 180, 50, 40, .99, 'green')
        for _ in range(3):
            tracker.update([a], (400, 600, 3))
        self.assertEqual(1, tracker.update([b], (400, 600, 3))[1])


class InputTests(unittest.TestCase):
    # All input primitives are mocked. These tests never click or type on the PC.
    def test_inactive_window_has_no_game_input(self):
        driver = win.Input()
        with patch.object(win, 'foreground', return_value=False), \
             patch.object(driver, '_mouse') as mouse, patch.object(driver, '_key') as key:
            self.assertFalse(driver.attack(123))
            self.assertFalse(driver.aim(123, 30, 10, True))
            self.assertFalse(driver.walk_pulse(123, threading.Event()))
            mouse.assert_not_called()
            key.assert_not_called()

    def test_walk_stop_releases_key(self):
        driver = win.Input()
        stop = threading.Event()
        stop.set()
        with patch.object(win, 'foreground', return_value=True), patch.object(driver, '_key') as key:
            driver.walk_pulse(123, stop)
            self.assertEqual([((0x57,), {}), ((0x57, True), {})], key.call_args_list)
            self.assertEqual(set(), driver.held)

    def test_camera_error_releases_button(self):
        driver = win.Input()
        with patch.object(win, 'foreground', return_value=True), patch.object(driver, '_center', return_value=True), \
             patch.object(driver, '_mouse', side_effect=[None, RuntimeError('test'), None]) as mouse:
            with self.assertRaises(RuntimeError):
                driver.aim(123, 30, 10, True)
            self.assertEqual(0x10, mouse.call_args_list[-1].args[0])
            self.assertFalse(driver.right)

    def test_windows_input_struct_layout(self):
        self.assertEqual(40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28, ctypes.sizeof(win.INPUT))


class EngineTests(unittest.TestCase):
    def run_fake(self, hunting=False, lose_focus=False, stop_key=False, blocked=None, enemies=True,
                 recover=False, first_fail=False, lose_target=False):
        events = queue.Queue()
        engine = Engine(events)
        windows = (win.Window(101, 1, 'account A'), win.Window(202, 2, 'account B'))
        engine.options = Options(windows, (), hunting=hunting, poll_interval=.01, auto_recover=recover)
        chosen, captures = [], []
        current = [999]
        failed = [False]

        def activate(hwnd):
            chosen.append(hwnd)
            current[0] = hwnd
            return True

        class Screen:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def grab(self, rect):
                captures.append(rect['left'])
                if first_fail and not failed[0]:
                    failed[0] = True
                    raise OSError('temporary capture failure')
                if len(captures) >= 9:
                    engine.stop()
                if lose_focus:
                    current[0] = 888
                return np.zeros((100, 150, 4), np.uint8)

        with patch('slime_assistant.engine.win.Input') as driver, \
             patch('slime_assistant.engine.win.available', return_value=True), \
             patch('slime_assistant.engine.win.activate', side_effect=activate), \
             patch('slime_assistant.engine.win.current_window', side_effect=lambda: current[0]), \
             patch('slime_assistant.engine.win.foreground', side_effect=lambda hwnd: current[0] == hwnd), \
             patch('slime_assistant.engine.win.capture_problem', side_effect=lambda hwnd, roi:
                   '가림' if hwnd == blocked and (not recover or current[0] != hwnd) else None), \
             patch('slime_assistant.engine.win.key_down', return_value=stop_key), \
             patch('slime_assistant.engine.win.client_rect', side_effect=lambda hwnd: {'left': hwnd}), \
             patch('slime_assistant.engine.Detector') as detector, \
             patch('slime_assistant.engine.mss.mss', return_value=Screen()):
            def detect(*args):
                if lose_target and len(captures) % 4 == 0:
                    return []
                height = 8 if lose_target else 20
                return [Detection(65, 50-height//2, 20, height, .99, 'slime')] if enemies else []
            detector.return_value.detect.side_effect = detect
            engine._run()
            if not hunting or lose_focus or stop_key or not enemies:
                for action in ('attack', 'aim', 'walk_pulse'):
                    getattr(driver.return_value, action).assert_not_called()
            driver.return_value.release.assert_called()
            attacks = [call.args[0] for call in driver.return_value.attack.call_args_list]
            engine.test_walk_calls = driver.return_value.walk_pulse.call_count
        return chosen, captures, attacks, list(events.queue), engine

    def test_detect_mode_monitors_both_without_activating_either(self):
        chosen, captures, attacks, events, _ = self.run_fake()
        self.assertEqual([], chosen)
        self.assertEqual([101, 202, 101, 202], captures[:4])
        self.assertEqual('stopped', events[-1][0])

    def test_focus_loss_stops_before_input(self):
        chosen, _, _, events, _ = self.run_fake(hunting=True, lose_focus=True)
        self.assertEqual([], chosen)
        self.assertTrue(any(kind == 'error' for kind, _ in events))

    def test_emergency_key_stops_before_input(self):
        chosen, captures, _, events, _ = self.run_fake(hunting=True, stop_key=True)
        self.assertEqual([], chosen)
        self.assertEqual([], captures)
        self.assertFalse(any(kind == 'error' for kind, _ in events))

    def test_both_accounts_confirm_and_attack_separately(self):
        chosen, captures, attacks, events, engine = self.run_fake(hunting=True)
        self.assertEqual([101, 202], chosen)
        self.assertEqual([101, 202], attacks)
        self.assertEqual({101: 1, 202: 1}, engine.frames.get_nowait()[3])
        self.assertFalse(any(kind == 'error' for kind, _ in events), events)

    def test_no_target_never_activates_windows(self):
        chosen, _, attacks, _, _ = self.run_fake(hunting=True, enemies=False)
        self.assertEqual([], chosen)
        self.assertEqual([], attacks)

    def test_covered_account_is_not_captured_or_activated(self):
        chosen, captures, attacks, events, _ = self.run_fake(hunting=True, blocked=101)
        self.assertNotIn(101, captures)
        self.assertNotIn(101, chosen)
        self.assertNotIn(101, attacks)
        self.assertTrue(any(kind == 'monitor' and data[1] == '가림' for kind, data in events))

    def test_recovery_reveals_account_and_confirms_before_attacking(self):
        chosen, captures, attacks, events, _ = self.run_fake(hunting=True, blocked=101, recover=True)
        self.assertEqual(101, chosen[0])
        self.assertIn(101, captures)
        self.assertIn(101, attacks)
        self.assertFalse(any(kind == 'error' for kind, _ in events), events)

    def test_detect_only_never_reveals_covered_windows(self):
        chosen, captures, attacks, events, _ = self.run_fake(blocked=101, recover=True)
        self.assertEqual([], chosen)
        self.assertNotIn(101, captures)

    def test_temporary_capture_error_keeps_other_account_running(self):
        chosen, captures, attacks, events, _ = self.run_fake(hunting=True, recover=True, first_fail=True)
        self.assertIn(202, attacks)
        self.assertTrue(any(kind == 'log' and 'temporary capture failure' in data for kind, data in events))
        self.assertFalse(any(kind == 'error' for kind, _ in events), events)

    def test_lost_target_does_not_walk_on_old_coordinates(self):
        _, _, _, events, engine = self.run_fake(hunting=True, lose_target=True)
        self.assertEqual(0, engine.test_walk_calls)
        self.assertFalse(any(kind == 'error' for kind, _ in events), events)


class ChaseTests(unittest.TestCase):
    def test_first_person_is_default_without_right_button(self):
        self.assertFalse(Options((), ()).right_aim)

    def test_turn_then_approach_then_attack_and_stop_on_loss(self):
        engine = Engine(queue.Queue())
        engine.options = Options((), (), hunting=True)
        state = {'seen': 10, 'attack_time': 0., 'scan_time': 0.}
        side = Detection(110, 40, 20, 20, .99, 'slime')
        far = Detection(65, 46, 20, 8, .99, 'slime')
        near = Detection(65, 40, 20, 20, .99, 'slime')
        self.assertEqual('aim', engine.action_for(side, 3, (100, 150, 3), state, 10)[0])
        self.assertEqual(('walk',), engine.action_for(far, 3, (100, 150, 3), state, 10))
        self.assertEqual(('attack',), engine.action_for(near, 3, (100, 150, 3), state, 10))
        self.assertIsNone(engine.action_for(near, 2, (100, 150, 3), state, 10))
        self.assertIsNone(engine.action_for(None, 0, (100, 150, 3), state, 10))


class LayoutTests(unittest.TestCase):
    def test_tiles_fit_without_overlap(self):
        for count in (1, 2, 3, 4, 6, 9):
            rects = win.tile_rects(count, (0, 0, 1920, 1040))
            self.assertEqual(count, len(rects))
            for n, (x, y, w, h) in enumerate(rects):
                self.assertLessEqual(x + w, 1920)
                self.assertLessEqual(y + h, 1040)
                for a, b, c, d in rects[n+1:]:
                    self.assertTrue(x + w <= a or a + c <= x or y + h <= b or b + d <= y)


if __name__ == '__main__':
    unittest.main()
