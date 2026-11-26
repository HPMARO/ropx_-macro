from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk

from . import windows as win
from .engine import Engine, Options
from .vision import Detector, Template, Tracker, preview


def data_directory():
    base = os.environ.get('LOCALAPPDATA', str(Path.home()))
    root = Path(base) / 'SlimeAssistant'
    (root / 'templates').mkdir(parents=True, exist_ok=True)
    return root


class CropDialog(tk.Toplevel):
    def __init__(self, master, frame, title, on_crop, min_size=12):
        super().__init__(master)
        self.title(title)
        self.frame_image = frame
        self.on_crop = on_crop
        self.min_size = min_size
        self.start = None
        self.box = None
        self.selection = None
        h, w = frame.shape[:2]
        self.scale = min(1., 1080 / w, 640 / h,
                         (self.winfo_screenwidth() - 120) / w,
                         (self.winfo_screenheight() - 180) / h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pic = Image.fromarray(rgb).resize((round(w * self.scale), round(h * self.scale)))
        self.photo = ImageTk.PhotoImage(pic)
        ttk.Label(self, text='마우스로 드래그해 영역을 선택한 뒤 [선택 적용]을 누르세요.').pack(padx=12, pady=10)
        self.canvas = tk.Canvas(self, width=pic.width, height=pic.height, cursor='crosshair', highlightthickness=0)
        self.canvas.pack(padx=10)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.canvas.bind('<ButtonPress-1>', self.begin)
        self.canvas.bind('<B1-Motion>', self.move)
        self.canvas.bind('<ButtonRelease-1>', self.finish)
        row = ttk.Frame(self)
        row.pack(fill='x', padx=12, pady=10)
        self.caption = ttk.Label(row, text='선택 없음')
        self.caption.pack(side='left')
        ttk.Button(row, text='취소', command=self.destroy).pack(side='right')
        ttk.Button(row, text='선택 적용', command=self.apply).pack(side='right', padx=8)
        self.transient(master)
        self.grab_set()

    def point(self, event):
        return (max(0, min(self.photo.width() - 1, event.x)),
                max(0, min(self.photo.height() - 1, event.y)))

    def begin(self, event):
        self.start = self.point(event)
        if self.box:
            self.canvas.delete(self.box)
        x, y = self.start
        self.box = self.canvas.create_rectangle(x, y, x, y, outline='#ffdc39', width=3)

    def move(self, event):
        if self.start:
            self.canvas.coords(self.box, *self.start, *self.point(event))

    def finish(self, event):
        if self.start:
            x, y = self.start
            ex, ey = self.point(event)
            self.selection = tuple(round(v / self.scale) for v in (min(x, ex), min(y, ey), max(x, ex), max(y, ey)))
            x1, y1, x2, y2 = self.selection
            self.caption.config(text=f'{x2 - x1} × {y2 - y1} 픽셀')

    def apply(self):
        if not self.selection:
            return
        x1, y1, x2, y2 = self.selection
        if min(x2 - x1, y2 - y1) < self.min_size:
            messagebox.showwarning('영역 선택', f'가로·세로 {self.min_size}픽셀 이상 선택하세요.', parent=self)
            return
        try:
            self.on_crop(self.selection, self.frame_image)
        except Exception as exc:
            messagebox.showerror('등록 실패', str(exc), parent=self)
            return
        self.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title('Slime Assistant — 다계정 자동 추적·복구')
        root.geometry('1160x850')
        root.minsize(1000, 720)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.data_dir = data_directory()
        self.logger = logging.getLogger('slime-assistant')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(RotatingFileHandler(self.data_dir / 'activity.log', maxBytes=1_000_000,
                                                    backupCount=2, encoding='utf-8'))
        self.events = queue.Queue()
        self.engine = Engine(self.events)
        self.windows = {}
        self.checked = set()
        self.aliases = {}
        self.templates = []
        self.last_frame = None
        self.capture_busy = False
        self.offline_image = None
        self.hotkey_state = {0x77: False, 0x78: False}
        self.running = False
        self.start_after_id = None
        self.roi = (.08, .10, .88, .74)
        self.threshold = tk.DoubleVar(value=.84)
        self.poll_interval = tk.DoubleVar(value=.25)
        self.minimize_monitor = tk.BooleanVar(value=True)
        self.near = tk.DoubleVar(value=.16)
        self.gain = tk.DoubleVar(value=.20)
        self.interval = tk.DoubleVar(value=.65)
        self.mode = tk.StringVar(value='감지 전용 (입력 없음)')
        self.approach = tk.BooleanVar(value=True)
        self.auto_recover = tk.BooleanVar(value=True)
        self.scan = tk.BooleanVar(value=False)
        self.right_aim = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='정지 — 창 선택 → 기준 이미지 등록 → 감지 전용으로 확인')
        self.detail = tk.StringVar(value='슬라임 종류마다 근거리·중거리 이미지를 여러 장 등록하면 인식에 도움이 됩니다.')
        self.build_ui()
        self.load_settings()
        self.load_templates()
        self.refresh_windows()
        self.root.after(100, self.pump)
        self.log('준비 완료. F8 시작/정지 · F9 긴급 정지. 게임 입력은 사냥 모드에서만 발생합니다.')

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='SLIME ASSISTANT', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(outer, text='사냥 모드는 몹 추적·접근·공격과 가림 복구를 수행합니다. 실제 화면·마우스·키보드를 사용합니다.',
                  foreground='#9b4200').pack(anchor='w', pady=(2, 10))
        upper = ttk.Frame(outer)
        upper.pack(fill='x')
        left = ttk.LabelFrame(upper, text='1. 대상 Roblox 창 — 행 클릭으로 선택', padding=8)
        left.pack(side='left', fill='both', expand=True, padx=(0, 8))
        self.table = ttk.Treeview(left, columns=('pick', 'alias', 'pid', 'title', 'state'), show='headings', height=5)
        for col, title, width in [('pick', '선택', 40), ('alias', '구분 이름', 90), ('pid', 'PID', 65), ('title', '창 제목', 110), ('state', '감시 상태', 190)]:
            self.table.heading(col, text=title)
            self.table.column(col, width=width, stretch=col == 'title')
        self.table.pack(fill='x', expand=True)
        self.table.bind('<ButtonRelease-1>', self.toggle_window)
        bar = ttk.Frame(left)
        bar.pack(fill='x', pady=(6, 0))
        ttk.Button(bar, text='창 새로고침', command=self.refresh_windows).pack(side='left')
        ttk.Button(bar, text='계정 구분 이름', command=self.rename_window).pack(side='left', padx=6)
        ttk.Button(bar, text='선택 창 촬영', command=self.capture_window).pack(side='left')
        ttk.Button(bar, text='작은 창 자동 배치', command=self.tile_windows).pack(side='left', padx=6)
        right = ttk.LabelFrame(upper, text='2. 슬라임 기준 이미지', padding=8)
        right.pack(side='left', fill='both')
        self.template_list = tk.Listbox(right, height=5, width=28, exportselection=False)
        self.template_list.pack(fill='both', expand=True)
        bar = ttk.Frame(right)
        bar.pack(fill='x', pady=(6, 0))
        ttk.Button(bar, text='사진 열기', command=self.open_image).pack(side='left')
        ttk.Button(bar, text='슬라임 자르기', command=self.crop_template).pack(side='left', padx=4)
        ttk.Button(bar, text='기준 삭제', command=self.delete_template).pack(side='left')

        controls = ttk.LabelFrame(outer, text='3. 실행 설정', padding=8)
        controls.pack(fill='x', pady=8)
        row = ttk.Frame(controls)
        row.pack(fill='x')
        ttk.Combobox(row, textvariable=self.mode, values=['감지 전용 (입력 없음)', '사냥 모드 (실제 입력)'],
                     state='readonly', width=24).pack(side='left')
        for title, variable, start, end, increment in [
            ('인식 기준', self.threshold, .60, .99, .01), ('감시 간격', self.poll_interval, .1, 3, .05),
            ('근접 높이비', self.near, .03, .5, .01), ('시점 감도', self.gain, .02, .8, .02),
            ('공격 간격', self.interval, .2, 3, .05)]:
            ttk.Label(row, text=title).pack(side='left', padx=(10, 3))
            ttk.Spinbox(row, textvariable=variable, from_=start, to=end, increment=increment, width=5).pack(side='left')
        row = ttk.Frame(controls)
        row.pack(fill='x', pady=(8, 0))
        ttk.Checkbutton(row, text='몹 추적·접근 W', variable=self.approach).pack(side='left')
        ttk.Checkbutton(row, text='미감지 시 회전 탐색 (실험)', variable=self.scan).pack(side='left', padx=8)
        ttk.Checkbutton(row, text='우클릭 시점 조정', variable=self.right_aim).pack(side='left')
        ttk.Checkbutton(row, text='시작 시 도우미 최소화', variable=self.minimize_monitor).pack(side='left', padx=6)
        row = ttk.Frame(controls)
        row.pack(fill='x', pady=(5, 0))
        ttk.Checkbutton(row, text='가림·일시 오류 자동 복구 (사냥 중 창을 다시 가져옴)', variable=self.auto_recover).pack(side='left')
        ttk.Button(row, text='인식 영역 지정', command=self.crop_roi).pack(side='right')
        ttk.Button(row, text='설정 저장', command=self.save_settings).pack(side='right', padx=8)
        row = ttk.Frame(outer)
        row.pack(fill='x', pady=(0, 6))
        self.start_button = ttk.Button(row, text='시작 / 정지 (F8)', command=self.toggle)
        self.start_button.pack(side='left')
        ttk.Button(row, text='즉시 정지 (F9)', command=self.stop).pack(side='left', padx=8)
        ttk.Button(row, text='현재 사진 인식 시험', command=self.test_image).pack(side='left')
        ttk.Label(row, textvariable=self.status).pack(side='left', padx=12)
        self.preview_label = ttk.Label(outer, anchor='center', background='#17212b')
        self.preview_label.pack(fill='both', expand=True)
        ttk.Label(outer, textvariable=self.detail, wraplength=1080).pack(anchor='w', pady=5)
        self.logs = tk.Text(outer, height=5, state='disabled', font=('Consolas', 9))
        self.logs.pack(fill='x')

    def log(self, text):
        self.logger.info(text)
        self.logs.config(state='normal')
        self.logs.insert('end', time.strftime('%H:%M:%S ') + text + '\n')
        if int(self.logs.index('end-1c').split('.')[0]) > 300:
            self.logs.delete('1.0', '100.0')
        self.logs.see('end')
        self.logs.config(state='disabled')

    def refresh_windows(self):
        if self.running:
            return
        try:
            current = win.windows()
            self.windows = {str(w.hwnd): w for w in current}
            self.checked.intersection_update(self.windows)
            self.aliases = {k: v for k, v in self.aliases.items() if k in self.windows}
            self.table.delete(*self.table.get_children())
            for n, (key, w) in enumerate(self.windows.items(), 1):
                self.aliases.setdefault(key, f'계정 {n}')
                self.table.insert('', 'end', iid=key, values=('✓' if key in self.checked else '',
                                                            self.aliases[key], w.pid, w.title, '대기'))
            if not current:
                self.status.set('Roblox 실행 창 없음 — 게임 접속 후 새로고침')
        except Exception as exc:
            self.log(f'창 검색 오류: {exc}')

    def toggle_window(self, event):
        if self.running:
            return
        key = self.table.identify_row(event.y)
        if key:
            self.checked.symmetric_difference_update({key})
            self.table.set(key, 'pick', '✓' if key in self.checked else '')

    def tile_windows(self):
        if self.running or self.engine.running:
            return
        targets = [w for key, w in self.windows.items() if key in self.checked]
        if not targets:
            messagebox.showinfo('작은 창 배치', '배치할 Roblox 창을 먼저 체크하세요.')
            return
        try:
            win.tile_windows(targets)
            self.log('선택한 게임 창을 배치했습니다. 이 크기에서 슬라임 기준을 등록하세요.')
            self.status.set('배치 완료 — 도우미가 게임을 가리면 감시 시작 시 최소화하세요.')
        except Exception as exc:
            messagebox.showerror('창 배치', str(exc))

    def selected_window(self):
        selected = self.table.selection()
        if selected and selected[0] in self.windows:
            return self.windows[selected[0]]
        if self.checked:
            return self.windows[next(k for k in self.windows if k in self.checked)]
        raise ValueError('목록에서 대상 게임 창을 먼저 선택하세요.')

    def rename_window(self):
        if self.running:
            return
        try:
            w = self.selected_window()
            name = simpledialog.askstring('계정 구분', '비밀번호가 아닌 구분용 이름을 입력하세요.',
                                          initialvalue=self.aliases[str(w.hwnd)], parent=self.root)
            if name:
                self.aliases[str(w.hwnd)] = name[:40]
                self.table.set(str(w.hwnd), 'alias', name[:40])
        except ValueError as exc:
            messagebox.showinfo('창 선택', str(exc))

    def capture_window(self):
        if self.running or self.capture_busy:
            messagebox.showinfo('촬영', '실행을 정지한 뒤 촬영하세요.')
            return
        try:
            w = self.selected_window()
            if not win.available(w.hwnd):
                raise ValueError('대상 게임 창을 최소화 해제하세요.')
            self.capture_busy = True
            self.root.withdraw()
            if not win.activate(w.hwnd):
                raise ValueError('게임 창을 활성화할 수 없습니다. 게임 화면을 직접 캡처한 사진을 열어도 됩니다.')
            self.root.after(1000, lambda: self.finish_capture(w))
        except Exception as exc:
            self.capture_busy = False
            self.root.deiconify()
            messagebox.showerror('촬영 실패', str(exc))

    def finish_capture(self, w):
        try:
            if not win.foreground(w.hwnd):
                raise ValueError('촬영 전에 대상 창이 바뀌었습니다. 다시 촬영하세요.')
            with mss.mss() as sct:
                self.offline_image = np.array(sct.grab(win.client_rect(w.hwnd)))[:, :, :3].copy()
            self.last_frame = self.offline_image
            self.show_frame(self.offline_image)
            self.log(f'게임 화면 촬영 완료 / PID {w.pid}. [슬라임 자르기]로 등록하세요.')
        except Exception as exc:
            messagebox.showerror('촬영 실패', str(exc), parent=self.root)
        finally:
            self.capture_busy = False
            self.root.deiconify()
            self.root.lift()

    def open_image(self):
        if self.running:
            return
        name = filedialog.askopenfilename(title='슬라임이 보이는 게임 사진',
                                          filetypes=[('사진', '*.png *.jpg *.jpeg *.bmp'), ('모든 파일', '*.*')])
        if not name:
            return
        frame = cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror('사진', '이미지를 읽지 못했습니다.')
            return
        if max(frame.shape[:2]) > 10000:
            messagebox.showerror('사진', '너무 큰 사진입니다. 게임 스크린샷을 사용하세요.')
            return
        self.offline_image = frame
        self.last_frame = frame
        self.show_frame(frame)
        self.log('사진 불러오기 완료. 슬라임 몸체를 드래그해 기준으로 등록하세요.')

    def require_frame(self):
        if self.running:
            messagebox.showinfo('정지 필요', 'F9로 정지한 뒤 등록·영역 설정을 하세요.')
            return None
        if self.last_frame is None:
            messagebox.showinfo('사진 필요', '[선택 창 촬영] 또는 [사진 열기]를 먼저 사용하세요.')
            return None
        return self.last_frame

    def crop_template(self):
        frame = self.require_frame()
        if frame is not None:
            CropDialog(self.root, frame, '슬라임 등록 — 몸체 전체, 배경·UI는 최소한', self.add_template)

    def add_template(self, rect, frame):
        x1, y1, x2, y2 = rect
        crop = frame[y1:y2, x1:x2].copy()
        Template('검증', crop)
        name = simpledialog.askstring('슬라임 이름', '예: 파랑_중거리_정면 (계정 공통 사용)', parent=self.root)
        if not name:
            return
        name = re.sub(r'[^\w가-힣-]', '_', name)[:45] or 'slime'
        path = self.data_dir / 'templates' / f'{name}_{time.time_ns()}.png'
        ok, encoded = cv2.imencode('.png', crop)
        if not ok:
            raise ValueError('이미지 저장 실패')
        encoded.tofile(str(path))
        self.load_templates()
        self.log(f'기준 등록: {name}. 색·거리·각도가 다른 모습은 별도로 등록하세요.')

    def crop_roi(self):
        frame = self.require_frame()
        if frame is not None:
            CropDialog(self.root, frame, '인식 영역 — 채팅·체력·아이템 UI를 제외한 게임 영역', self.set_roi, min_size=50)

    def set_roi(self, rect, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = rect
        self.roi = (x1 / w, y1 / h, x2 / w, y2 / h)
        self.save_settings()
        self.test_image()

    def load_templates(self):
        self.templates = []
        self.template_paths = []
        self.template_list.delete(0, 'end')
        for path in sorted((self.data_dir / 'templates').glob('*.png')):
            try:
                self.templates.append(Template.load(path))
                self.template_paths.append(path)
                self.template_list.insert('end', path.stem.rsplit('_', 1)[0])
            except Exception as exc:
                self.log(f'기준 이미지 제외: {path.name}: {exc}')

    def delete_template(self):
        if self.running:
            return
        selected = self.template_list.curselection()
        if not selected:
            return
        path = self.template_paths[selected[0]]
        # Recoverable local deletion only: move to a sibling archive, never remove user screenshots.
        archive = self.data_dir / 'archived-templates'
        archive.mkdir(exist_ok=True)
        path.rename(archive / path.name)
        self.load_templates()
        self.log('선택한 기준을 archived-templates 폴더로 이동했습니다.')

    def checked_options(self):
        targets = tuple(w for key, w in self.windows.items() if key in self.checked)
        if not targets:
            raise ValueError('사용할 Roblox 창을 체크하세요.')
        if not self.templates:
            raise ValueError('슬라임 기준 이미지를 한 장 이상 등록하세요.')
        threshold, poll_interval, near, gain, interval = (float(v.get()) for v in
                                                   (self.threshold, self.poll_interval, self.near, self.gain, self.interval))
        if not (.6 <= threshold <= .99 and .1 <= poll_interval <= 3 and .03 <= near <= .5
                and .02 <= gain <= .8 and .2 <= interval <= 3):
            raise ValueError('설정값이 허용 범위를 벗어났습니다.')
        return Options(targets, tuple(self.templates), self.roi, threshold,
                       self.mode.get().startswith('사냥'), poll_interval, self.approach.get(), self.scan.get(),
                       self.right_aim.get(), near, gain, interval, self.auto_recover.get())

    def toggle(self):
        if self.running:
            self.stop()
            return
        if self.capture_busy or self.engine.running:
            return
        try:
            options = self.checked_options()
        except Exception as exc:
            messagebox.showwarning('시작 전 확인', str(exc))
            return
        self.save_settings()
        self.running = True
        self.status.set('2초 후 시작 — F9 정지 / 게임에서 무기를 미리 장착하세요')
        self.log(f'시작 예약: {len(options.targets)}개 창 / {"사냥" if options.hunting else "감지 전용"}')
        self.start_after_id = self.root.after(2000, lambda: self.begin(options))

    def begin(self, options):
        self.start_after_id = None
        if self.running and self.minimize_monitor.get():
            self.root.iconify()
            self.start_after_id = self.root.after(150, lambda: self.launch(options))
        else:
            self.launch(options)

    def launch(self, options):
        self.start_after_id = None
        if self.running:
            try:
                self.engine.start(options)
            except Exception as exc:
                self.running = False
                self.log(str(exc))

    def stop(self):
        if self.start_after_id is not None:
            self.root.after_cancel(self.start_after_id)
            self.start_after_id = None
        self.running = False
        self.engine.stop()
        self.status.set('정지 요청 — 누른 키 해제 중')

    def test_image(self):
        if self.running or self.last_frame is None:
            return
        if not self.templates:
            messagebox.showinfo('인식 시험', '기준 이미지를 먼저 등록하세요.')
            return
        frame = self.last_frame.copy()
        try:
            detections = Detector(self.templates).detect(frame, self.roi, float(self.threshold.get()))
            target, _ = Tracker().update(detections, frame.shape)
            self.show_frame(preview(frame, detections, target, self.roi))
            self.detail.set(f'사진 시험: {len(detections)}개 감지. 노란 박스=우선 표적, 녹색=후보. 실제 게임 입력 없음.')
            self.log(f'사진 인식 시험: 후보 {len(detections)}개')
        except Exception as exc:
            messagebox.showerror('인식 오류', str(exc))

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        width = max(500, self.preview_label.winfo_width())
        height = max(200, self.preview_label.winfo_height())
        im.thumbnail((width, height))
        self.photo = ImageTk.PhotoImage(im)
        self.preview_label.config(image=self.photo)

    def settings(self):
        return {'roi': list(self.roi), 'threshold': float(self.threshold.get()), 'poll_interval': float(self.poll_interval.get()),
                'near': float(self.near.get()), 'gain': float(self.gain.get()), 'interval': float(self.interval.get()),
                'chase_enabled': self.approach.get(), 'auto_recover': self.auto_recover.get(),
                'scan': self.scan.get(), 'right_aim': self.right_aim.get(),
                'minimize_monitor': self.minimize_monitor.get()}

    def save_settings(self):
        try:
            content = json.dumps(self.settings(), ensure_ascii=False, indent=2)
            temp = self.data_dir / 'settings.pending'
            temp.write_text(content, encoding='utf-8')
            temp.replace(self.data_dir / 'settings.json')
        except Exception as exc:
            self.log(f'설정 저장 실패: {exc}')

    def load_settings(self):
        path = self.data_dir / 'settings.json'
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            roi = data.get('roi', self.roi)
            if len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1:
                self.roi = tuple(roi)
            for key, variable in [('threshold', self.threshold), ('poll_interval', self.poll_interval), ('near', self.near),
                                  ('gain', self.gain), ('interval', self.interval)]:
                value = data.get(key)
                if isinstance(value, (float, int)):
                    variable.set(value)
            for key, variable in [('chase_enabled', self.approach), ('auto_recover', self.auto_recover),
                                  ('scan', self.scan), ('right_aim', self.right_aim),
                                  ('minimize_monitor', self.minimize_monitor)]:
                if isinstance(data.get(key), bool):
                    variable.set(data[key])
            # Never persist an armed state or automatically reuse window handles.
        except Exception as exc:
            self.log(f'이전 설정을 읽지 못했습니다: {exc}')

    def pump(self):
        for vk in self.hotkey_state:
            pressed = win.key_down(vk)
            if pressed and not self.hotkey_state[vk]:
                if vk == 0x78:
                    self.stop()
                elif self.root.grab_current() is None and not self.capture_busy:
                    self.toggle()
            self.hotkey_state[vk] = pressed
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                self.log(data)
            elif kind == 'error':
                self.log(f'중지: {data}')
                self.detail.set(data)
            elif kind == 'active':
                self.status.set(f'{self.aliases.get(str(data.hwnd), data.title)} / PID {data.pid}')
            elif kind == 'monitor':
                window, state = data
                key = str(window.hwnd)
                if self.table.exists(key):
                    self.table.set(key, 'state', state)
            elif kind == 'stopped':
                self.running = False
                self.status.set('정지 — F8로 다시 시작')
                self.log('실행 종료. 보유 키·마우스 버튼 해제를 시도했습니다.')
        try:
            w, frame, status, counts, age = self.engine.frames.get_nowait()
            # Keep the raw capture separate from annotations for future cropping.
            self.show_frame(frame)
            attacks = ' / '.join(f'{self.aliases.get(str(k), str(k))}: {v}회' for k, v in counts.items())
            if self.running:
                self.status.set(status)
            self.detail.set(f'{status} | 공격 입력: {attacks} (처치 수가 아님)')
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def close(self):
        self.stop()
        self.save_settings()
        self.wait_close()

    def wait_close(self):
        if self.engine.running:
            self.root.after(100, self.wait_close)
        else:
            self.root.destroy()


def main():
    if os.name != 'nt':
        raise SystemExit('Windows 10/11 전용 프로그램입니다.')
    win.dpi_aware()
    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista')
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
