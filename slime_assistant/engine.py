from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time

import mss
import numpy as np

from .vision import Detector, Tracker, preview, dashboard
from . import windows as win


@dataclass(frozen=True)
class Options:
    targets: tuple
    templates: tuple
    roi: tuple = (.08, .10, .88, .74)
    threshold: float = .84
    hunting: bool = False
    poll_interval: float = .25
    auto_approach: bool = True
    auto_scan: bool = False
    right_aim: bool = True
    near_ratio: float = .16
    camera_gain: float = .20
    attack_interval: float = .65
    auto_recover: bool = True


class ObservationError(RuntimeError):
    pass


class Engine:
    def __init__(self, events):
        self.events = events
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = None
        self.options = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, options):
        if self.running:
            raise RuntimeError('이미 실행 중입니다.')
        if not options.targets:
            raise ValueError('대상 창이 없습니다.')
        self.options = options
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def put_frame(self, value):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(value)

    def interrupted(self):
        if win.key_down(0x78):
            self.stop_event.set()
        return self.stop_event.is_set()

    def action_for(self, target, hits, shape, state, now):
        o = self.options
        if target and hits >= 3:
            h, w = shape[:2]
            tx, ty = target.center
            ex, ey = tx - w / 2, ty - h / 2
            if abs(ex) > w * .07 or abs(ey) > h * .12:
                return ('aim', max(-45, min(45, round(ex * o.camera_gain))),
                        max(-35, min(35, round(ey * o.camera_gain))))
            if target.h / h < o.near_ratio:
                return ('walk',) if o.auto_approach else None
            if now - state['attack_time'] >= o.attack_interval:
                return ('attack',)
        if not target and o.auto_scan and now - state['seen'] > 2 and now - state['scan_time'] > .5:
            return ('scan', 35, 0)
        return None

    def observe(self, screen, detector, window, state):
        o = self.options
        if self.interrupted():
            return None
        if not win.available(window.hwnd, window.pid):
            raise ObservationError('닫힘·최소화: 창을 다시 열면 확인을 재개합니다.')
        if win.capture_problem(window.hwnd, o.roi):
            raise ObservationError('아직 가려져 있거나 화면 밖에 있습니다.')
        capture_time = time.monotonic()
        rect = win.client_rect(window.hwnd)
        frame = np.array(screen.grab(rect))[:, :, :3].copy()
        if state['shape'] != frame.shape or capture_time - state['observed'] > 1.5:
            state['tracker'].reset()
            state['shape'] = frame.shape
        matches = detector.detect(frame, o.roi, o.threshold)
        if (not win.available(window.hwnd, window.pid) or win.client_rect(window.hwnd) != rect
                or win.capture_problem(window.hwnd, o.roi)):
            raise ObservationError('관측 중 창이 가려지거나 위치가 바뀌었습니다.')
        if time.monotonic() - capture_time > .8:
            raise ObservationError('인식 지연: 기준 이미지 수·화면 크기를 줄여 보세요.')
        target, hits = state['tracker'].update(matches, frame.shape)
        state['observed'] = time.monotonic()
        if target:
            state['seen'] = time.monotonic()
        return frame, matches, target, hits

    def activate(self, window, driver):
        if self.interrupted():
            return False
        driver.release()
        if not win.available(window.hwnd, window.pid) or not win.activate(window.hwnd):
            raise ObservationError('창 활성화 실패: 잠시 후 다시 시도합니다.')
        if self.stop_event.wait(.12) or self.interrupted():
            return False
        if not win.foreground(window.hwnd):
            raise ObservationError('활성화 직후 포커스가 바뀌었습니다.')
        return True

    def _run(self):
        o = self.options
        driver = win.Input()
        detector = Detector(o.templates)
        states = {w.hwnd: {'tracker': Tracker(), 'seen': time.monotonic(),
                          'observed': 0., 'attack_time': 0., 'scan_time': 0., 'shape': None,
                          'retry_at': 0., 'failures': 0} for w in o.targets}
        counts = {w.hwnd: 0 for w in o.targets}
        pictures, labels = {}, {}
        expected_focus = win.current_window()
        try:
            with mss.mss() as screen:
                while not self.interrupted():
                    cycle = time.monotonic()
                    live = 0
                    for window in o.targets:
                        if self.interrupted():
                            break
                        if o.hunting and not o.auto_recover and win.current_window() != expected_focus:
                            raise RuntimeError('사용자가 다른 창을 선택해 사냥을 정지했습니다.')
                        state = states[window.hwnd]
                        if time.monotonic() < state['retry_at']:
                            continue
                        if not win.available(window.hwnd, window.pid):
                            state['tracker'].reset()
                            state['retry_at'] = time.monotonic() + 1
                            pictures.pop(window.hwnd, None)
                            labels[window.hwnd] = 'WAIT FOR WINDOW'
                            self.events.put(('monitor', (window, '닫힘·최소화: 대기')))
                            continue
                        live += 1
                        try:
                            reason = win.capture_problem(window.hwnd, o.roi)
                            if reason:
                                if o.hunting and o.auto_recover:
                                    self.events.put(('monitor', (window, '가림 복구: 창 활성화')))
                                    if not self.activate(window, driver):
                                        break
                                    expected_focus = window.hwnd
                                else:
                                    state['tracker'].reset()
                                    pictures.pop(window.hwnd, None)
                                    labels[window.hwnd] = 'PAUSED'
                                    self.events.put(('monitor', (window, reason)))
                                    continue
                            observation = self.observe(screen, detector, window, state)
                            if observation is None:
                                break
                            # Confirm in a short burst so overlapping accounts do not keep
                            # resetting each other's tracker before reaching three frames.
                            if o.hunting:
                                for _ in range(2):
                                    if observation[2] is None or observation[3] >= 3 or self.interrupted():
                                        break
                                    if self.stop_event.wait(.08):
                                        break
                                    observation = self.observe(screen, detector, window, state)
                                    if observation is None:
                                        break
                            if observation is None or self.interrupted():
                                break
                            frame, matches, target, hits = observation
                            pictures[window.hwnd] = preview(frame, matches, target, o.roi)
                            labels[window.hwnd] = f'{len(matches)} found / {hits} confirmations'
                            self.events.put(('monitor', (window, f'감지 {len(matches)} / 확인 {hits}회')))
                            action = self.action_for(target, hits, frame.shape, state, time.monotonic())
                            if o.hunting and action:
                                if not o.auto_recover and win.current_window() != expected_focus:
                                    raise RuntimeError('포커스 변경으로 사냥을 정지했습니다.')
                                if not self.activate(window, driver):
                                    break
                                expected_focus = window.hwnd
                                fresh = self.observe(screen, detector, window, state)
                                if fresh is None or self.interrupted():
                                    break
                                fresh_frame, _, fresh_target, confirmations = fresh
                                action = self.action_for(fresh_target, confirmations, fresh_frame.shape, state, time.monotonic())
                                if not win.foreground(window.hwnd):
                                    raise ObservationError('입력 전 포커스 변경: 재확인 예정')
                                if action:
                                    kind = action[0]
                                    if kind in ('aim', 'scan'):
                                        driver.aim(window.hwnd, *action[1:], o.right_aim)
                                        state['tracker'].reset()
                                        if kind == 'scan':
                                            state['scan_time'] = time.monotonic()
                                        phase = '방향 조정' if kind == 'aim' else '회전 탐색'
                                    elif kind == 'walk':
                                        driver.walk_pulse(window.hwnd, self.stop_event)
                                        state['tracker'].reset()
                                        phase = '몹에게 접근'
                                    else:
                                        if driver.attack(window.hwnd):
                                            counts[window.hwnd] += 1
                                            state['attack_time'] = time.monotonic()
                                        phase = '공격'
                                    self.events.put(('monitor', (window, phase)))
                            state['failures'] = 0
                            state['retry_at'] = 0.
                        except (RuntimeError, OSError, ValueError) as exc:
                            if not o.auto_recover:
                                raise
                            driver.release()
                            state['tracker'].reset()
                            state['failures'] += 1
                            delay = min(8, 2 ** min(3, state['failures'] - 1))
                            state['retry_at'] = time.monotonic() + delay
                            pictures.pop(window.hwnd, None)
                            labels[window.hwnd] = f'RETRY IN {delay}s'
                            self.events.put(('monitor', (window, f'{delay}초 후 재시도: {exc}')))
                            self.events.put(('log', f'PID {window.pid}: {exc} / {delay}초 후 재시도'))
                    duration = time.monotonic() - cycle
                    self.put_frame((None, dashboard(o.targets, pictures, labels),
                                    f'{len(o.targets)}개 창 순회 · {duration:.2f}초 · F9 정지', dict(counts), duration))
                    self.stop_event.wait(max(.05 if live else .5, o.poll_interval - duration))
        except Exception as exc:
            self.events.put(('error', str(exc)))
        finally:
            try:
                driver.release()
            except Exception as exc:
                self.events.put(('log', f'입력 해제 확인 필요: {exc}'))
            self.stop_event.set()
            self.events.put(('stopped', None))
