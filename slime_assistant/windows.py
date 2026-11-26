from __future__ import annotations

import ctypes as c
from ctypes import wintypes as wt
from dataclasses import dataclass
import os
import math
import time


@dataclass(frozen=True)
class Window:
    hwnd: int
    pid: int
    title: str


if os.name == 'nt':
    user = c.WinDLL('user32', use_last_error=True)
    kernel = c.WinDLL('kernel32', use_last_error=True)
    user.GetForegroundWindow.restype = wt.HWND
    user.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, c.c_int]
    user.GetWindowTextLengthW.argtypes = [wt.HWND]
    user.GetWindowThreadProcessId.argtypes = [wt.HWND, c.POINTER(wt.DWORD)]
    user.GetClientRect.argtypes = [wt.HWND, c.POINTER(wt.RECT)]
    user.ClientToScreen.argtypes = [wt.HWND, c.POINTER(wt.POINT)]
    user.IsWindow.argtypes = [wt.HWND]
    user.IsIconic.argtypes = [wt.HWND]
    user.IsWindowVisible.argtypes = [wt.HWND]
    user.SetForegroundWindow.argtypes = [wt.HWND]
    user.SetCursorPos.argtypes = [c.c_int, c.c_int]
    user.WindowFromPoint.argtypes = [wt.POINT]
    user.WindowFromPoint.restype = wt.HWND
    user.GetAncestor.argtypes = [wt.HWND, wt.UINT]
    user.GetAncestor.restype = wt.HWND
    user.SetWindowPos.argtypes = [wt.HWND, wt.HWND, c.c_int, c.c_int, c.c_int, c.c_int, wt.UINT]
    user.ShowWindow.argtypes = [wt.HWND, c.c_int]
    user.SystemParametersInfoW.argtypes = [wt.UINT, wt.UINT, c.c_void_p, wt.UINT]
    user.GetAsyncKeyState.argtypes = [c.c_int]
    user.GetAsyncKeyState.restype = c.c_short
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, c.POINTER(wt.DWORD)]
    kernel.CloseHandle.argtypes = [wt.HANDLE]


def dpi_aware():
    if os.name == 'nt':
        try:
            user.SetProcessDpiAwarenessContext(c.c_void_p(-4))
        except (AttributeError, OSError):
            user.SetProcessDPIAware()


def windows():
    if os.name != 'nt':
        return []
    found = []
    callback_type = c.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    def visit(hwnd, _):
        if not user.IsWindowVisible(hwnd):
            return True
        size = user.GetWindowTextLengthW(hwnd)
        title = c.create_unicode_buffer(size + 1)
        user.GetWindowTextW(hwnd, title, size + 1)
        pid = wt.DWORD()
        user.GetWindowThreadProcessId(hwnd, c.byref(pid))
        process = kernel.OpenProcess(0x1000, False, pid.value)
        path = ''
        if process:
            buf = c.create_unicode_buffer(32768)
            length = wt.DWORD(len(buf))
            if kernel.QueryFullProcessImageNameW(process, 0, buf, c.byref(length)):
                path = buf.value
            kernel.CloseHandle(process)
        if 'robloxplayer' in path.lower() and title.value:
            found.append(Window(int(hwnd), pid.value, title.value))
        return True

    user.EnumWindows.argtypes = [callback_type, wt.LPARAM]
    user.EnumWindows(callback_type(visit), 0)
    return found


def foreground(hwnd):
    return os.name == 'nt' and current_window() == hwnd


def current_window():
    return int(user.GetForegroundWindow() or 0) if os.name == 'nt' else 0


def capture_problem(hwnd, roi):
    """Sample the ROI for covering windows; no focus changes or input."""
    rect = client_rect(hwnd)
    x1 = rect['left'] + round(rect['width'] * roi[0])
    y1 = rect['top'] + round(rect['height'] * roi[1])
    x2 = rect['left'] + round(rect['width'] * roi[2]) - 1
    y2 = rect['top'] + round(rect['height'] * roi[3]) - 1
    for row in range(5):
        for col in range(5):
            point = wt.POINT(round(x1 + (x2 - x1) * col / 4), round(y1 + (y2 - y1) * row / 4))
            hit = user.WindowFromPoint(point)
            root = user.GetAncestor(hit, 2) if hit else None
            if int(root or 0) != hwnd:
                return '탐색 영역 가림·화면 밖 (창 배치 확인)'
    return None


def tile_rects(count, area):
    if count < 1:
        return []
    left, top, width, height = area
    columns = min(count, max(1, math.ceil(math.sqrt(count * width / height / 1.6))))
    rows = math.ceil(count / columns)
    cell_w, cell_h = width // columns, height // rows
    if cell_w < 320 or cell_h < 240:
        raise ValueError('한 화면에 배치하기에 창이 너무 많습니다. 계정 수를 줄이거나 직접 배치하세요.')
    return [(left + (n % columns) * cell_w, top + (n // columns) * cell_h,
             cell_w, cell_h) for n in range(count)]


def tile_windows(targets):
    area = wt.RECT()
    if not user.SystemParametersInfoW(0x30, 0, c.byref(area), 0):
        raise RuntimeError('모니터 작업 영역을 읽지 못했습니다.')
    positions = tile_rects(len(targets), (area.left, area.top, area.right - area.left, area.bottom - area.top))
    for window, (x, y, w, h) in zip(targets, positions):
        if not user.IsWindow(window.hwnd):
            continue
        # SW_SHOWNOACTIVATE restores a minimized/maximized window without taking focus.
        user.ShowWindow(window.hwnd, 4)
        if not user.SetWindowPos(window.hwnd, None, x, y, w, h, 0x0014):
            raise RuntimeError('창 배치 실패. 창을 직접 배치해도 됩니다.')


def available(hwnd, pid=None):
    if os.name != 'nt' or not user.IsWindow(hwnd) or user.IsIconic(hwnd):
        return False
    if pid is not None:
        actual_pid = wt.DWORD()
        user.GetWindowThreadProcessId(hwnd, c.byref(actual_pid))
        return actual_pid.value == pid
    return True


def activate(hwnd):
    if not available(hwnd):
        return False
    if foreground(hwnd):
        return True
    user.SetForegroundWindow(hwnd)
    return foreground(hwnd)


def client_rect(hwnd):
    if not available(hwnd):
        raise RuntimeError('창이 닫혔거나 최소화돼 있습니다.')
    rect = wt.RECT()
    origin = wt.POINT(0, 0)
    if not user.GetClientRect(hwnd, c.byref(rect)) or not user.ClientToScreen(hwnd, c.byref(origin)):
        raise RuntimeError('게임 창의 위치를 읽지 못했습니다.')
    if rect.right < 100 or rect.bottom < 100:
        raise RuntimeError('게임 창이 너무 작습니다.')
    return {'left': origin.x, 'top': origin.y, 'width': rect.right, 'height': rect.bottom}


def key_down(vk):
    return os.name == 'nt' and bool(user.GetAsyncKeyState(vk) & 0x8000)


class MOUSEINPUT(c.Structure):
    _fields_ = [('dx', wt.LONG), ('dy', wt.LONG), ('mouseData', wt.DWORD),
                ('dwFlags', wt.DWORD), ('time', wt.DWORD), ('dwExtraInfo', c.c_size_t)]


class KEYBDINPUT(c.Structure):
    _fields_ = [('wVk', wt.WORD), ('wScan', wt.WORD), ('dwFlags', wt.DWORD),
                ('time', wt.DWORD), ('dwExtraInfo', c.c_size_t)]


class HARDWAREINPUT(c.Structure):
    _fields_ = [('uMsg', wt.DWORD), ('wParamL', wt.WORD), ('wParamH', wt.WORD)]


class INPUTUNION(c.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]


class INPUT(c.Structure):
    _anonymous_ = ('u',)
    _fields_ = [('type', wt.DWORD), ('u', INPUTUNION)]


class Input:
    """Foreground-only ordinary OS input, never attached to a game process."""

    def __init__(self):
        self.held = set()
        self.right = False
        if os.name == 'nt':
            user.SendInput.argtypes = [wt.UINT, c.POINTER(INPUT), c.c_int]
            user.SendInput.restype = wt.UINT

    def _send(self, event):
        if user.SendInput(1, c.byref(event), c.sizeof(INPUT)) != 1:
            raise RuntimeError('입력 전송 실패. 권한 차이 또는 Windows 입력 제한을 확인하세요.')

    def _key(self, vk, up=False):
        self._send(INPUT(type=1, ki=KEYBDINPUT(wVk=vk, dwFlags=2 if up else 0)))

    def _mouse(self, flags, dx=0, dy=0):
        self._send(INPUT(type=0, mi=MOUSEINPUT(dx=dx, dy=dy, dwFlags=flags)))

    def release(self):
        for vk in tuple(self.held):
            try:
                self._key(vk, True)
            finally:
                self.held.discard(vk)
        if self.right:
            try:
                self._mouse(0x0010)
            finally:
                self.right = False

    def aim(self, hwnd, dx, dy, right_button):
        if not foreground(hwnd):
            return False
        if abs(dx) + abs(dy) < 2:
            return True
        if not self._center(hwnd):
            return False
        if right_button:
            self._mouse(0x0008)
            self.right = True
        try:
            if foreground(hwnd):
                self._mouse(0x0001, dx, dy)
        finally:
            if self.right:
                self._mouse(0x0010)
                self.right = False
        return True

    def walk_pulse(self, hwnd, stop, seconds=.1):
        if not foreground(hwnd):
            return False
        self._key(0x57)
        self.held.add(0x57)
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if stop.wait(.01) or key_down(0x78) or not foreground(hwnd):
                    break
        finally:
            self._key(0x57, True)
            self.held.discard(0x57)
        return True

    def _center(self, hwnd):
        if not foreground(hwnd):
            return False
        rect = client_rect(hwnd)
        if not user.SetCursorPos(rect['left'] + rect['width'] // 2,
                                 rect['top'] + rect['height'] // 2):
            raise RuntimeError('마우스 위치를 설정하지 못했습니다.')
        if not foreground(hwnd):
            return False
        return True

    def attack(self, hwnd):
        if not self._center(hwnd):
            return False
        self._mouse(0x0002)
        try:
            pass
        finally:
            self._mouse(0x0004)
        return True
