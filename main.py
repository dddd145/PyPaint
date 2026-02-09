import sys
import cv2
import numpy as np
import typing
import time
import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui

class AdvancedImageApp(PySide6.QtWidgets.QMainWindow):
  def __init__(self) -> None:
    super().__init__()
    self.setWindowTitle("PyPainter")
    self.setMinimumSize(1200, 800)

    # --- 状態保持用の変数 ---
    self.raw_image: typing.Optional[np.ndarray] = None
    self.last_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.start_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.eraser_mode = False
    self.pressure_brush_mode = False
    self.current_bg_color = (255, 255, 255)
    self.current_brush_color = (0, 0, 0)
    self.zoom_factor = 1.0

    self.color_history: typing.List[typing.Tuple[int, int, int]] = [
        (0, 0, 0), (255, 255, 255)]
    self.max_history = 8

    self.draw_mode = 0
    self.selection_rect: typing.Optional[PySide6.QtCore.QRect] = None
    self.clipboard: typing.Optional[np.ndarray] = None
    self.is_modified = False
    self.last_time = 0.0
    self.current_velocity_size = 5.0

    self.is_panning = False
    self.last_mouse_pos = PySide6.QtCore.QPoint()

    self.undo_stack: typing.List[np.ndarray] = []
    self.redo_stack: typing.List[np.ndarray] = []
    self.max_undo = 30

    self.init_ui()
    self.create_blank_canvas(800, 600)

  def init_ui(self) -> None:
    central_widget = PySide6.QtWidgets.QWidget()
    self.setCentralWidget(central_widget)
    main_layout = PySide6.QtWidgets.QHBoxLayout(central_widget)

    # --- 左側サイドパネル（スクロール可能にする） ---
    side_scroll = PySide6.QtWidgets.QScrollArea()
    side_scroll.setFixedWidth(260)
    side_scroll.setWidgetResizable(True)
    side_scroll.setHorizontalScrollBarPolicy(
        PySide6.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    side_container = PySide6.QtWidgets.QWidget()
    side_panel = PySide6.QtWidgets.QVBoxLayout(side_container)
    side_panel.setContentsMargins(5, 5, 5, 5)
    side_panel.setSpacing(10)

    # 1. ナビゲーター
    nav_group = PySide6.QtWidgets.QGroupBox("ナビゲーター")
    nav_layout = PySide6.QtWidgets.QVBoxLayout()
    self.nav_label = PySide6.QtWidgets.QLabel()
    self.nav_label.setFixedSize(220, 160)
    self.nav_label.setAlignment(PySide6.QtCore.Qt.AlignmentFlag.AlignCenter)
    self.nav_label.setStyleSheet(
        "background-color: #000; border: 1px solid #555;")
    self.nav_label.mousePressEvent = self.navigator_mouse_event
    self.nav_label.mouseMoveEvent = self.navigator_mouse_event
    nav_layout.addWidget(self.nav_label)
    nav_group.setLayout(nav_layout)
    side_panel.addWidget(nav_group)

    # 2. ツール設定
    draw_group = PySide6.QtWidgets.QGroupBox("ツール設定")
    draw_layout = PySide6.QtWidgets.QVBoxLayout()

    self.brush_size_slider = self._create_slider(
        "ブラシサイズ", 1, 50, 5, draw_layout)

    self.mode_combo = PySide6.QtWidgets.QComboBox()
    self.mode_combo.addItems(
        ["通常ブラシ", "矩形 (Rect)", "円 (Circle)", "塗りつぶし (Fill)", "選択 (Select)"])
    self.mode_combo.currentIndexChanged.connect(self.change_draw_mode)
    draw_layout.addWidget(PySide6.QtWidgets.QLabel("描画モード:"))
    draw_layout.addWidget(self.mode_combo)

    self.brush_mode_btn = PySide6.QtWidgets.QPushButton("筆圧感度: OFF")
    self.brush_mode_btn.setCheckable(True)
    self.brush_mode_btn.clicked.connect(self.toggle_brush_mode)
    draw_layout.addWidget(self.brush_mode_btn)

    self.brush_color_btn = PySide6.QtWidgets.QPushButton("カラーピッカー")
    self.brush_color_btn.clicked.connect(self.change_brush_color)
    draw_layout.addWidget(self.brush_color_btn)

    draw_layout.addWidget(PySide6.QtWidgets.QLabel("カラー履歴:"))
    self.history_grid = PySide6.QtWidgets.QGridLayout()
    self.history_grid.setSpacing(4)
    draw_layout.addLayout(self.history_grid)
    self.update_history_ui()

    self.eraser_btn = PySide6.QtWidgets.QPushButton("消しゴムモード")
    self.eraser_btn.setCheckable(True)
    self.eraser_btn.clicked.connect(self.toggle_eraser)
    draw_layout.addWidget(self.eraser_btn)

    undo_redo_layout = PySide6.QtWidgets.QHBoxLayout()
    self.undo_btn = PySide6.QtWidgets.QPushButton("Undo")
    self.undo_btn.clicked.connect(self.undo)
    self.redo_btn = PySide6.QtWidgets.QPushButton("Redo")
    self.redo_btn.clicked.connect(self.redo)
    undo_redo_layout.addWidget(self.undo_btn)
    undo_redo_layout.addWidget(self.redo_btn)
    draw_layout.addLayout(undo_redo_layout)

    self.clear_btn = PySide6.QtWidgets.QPushButton("キャンバスクリア")
    self.clear_btn.clicked.connect(self.clear_canvas_with_undo)
    draw_layout.addWidget(self.clear_btn)

    draw_group.setLayout(draw_layout)
    side_panel.addWidget(draw_group)

    # 3. エフェクト
    proc_group = PySide6.QtWidgets.QGroupBox("エフェクト")
    proc_layout = PySide6.QtWidgets.QVBoxLayout()
    self.rotate_slider = self._create_slider(
        "回転", -180, 180, 0, proc_layout)
    self.blur_slider = self._create_slider("ぼかし", 1, 51, 1, proc_layout)
    self.effect_combo = PySide6.QtWidgets.QComboBox()
    self.effect_combo.addItems(["なし", "モノクロ", "セピア", "エッジ抽出"])
    self.effect_combo.currentIndexChanged.connect(self.apply_effects)
    proc_layout.addWidget(PySide6.QtWidgets.QLabel("フィルタ効果:"))
    proc_layout.addWidget(self.effect_combo)
    proc_group.setLayout(proc_layout)
    side_panel.addWidget(proc_group)

    # 4. キャンバス操作
    canvas_op_group = PySide6.QtWidgets.QGroupBox("画像編集")
    canvas_op_layout = PySide6.QtWidgets.QVBoxLayout()
    self.resize_btn = PySide6.QtWidgets.QPushButton("解像度リサイズ")
    self.resize_btn.clicked.connect(self.resize_image_dialog)
    canvas_op_layout.addWidget(self.resize_btn)
    self.crop_btn = PySide6.QtWidgets.QPushButton("選択範囲で切抜き")
    self.crop_btn.clicked.connect(self.crop_to_selection)
    canvas_op_layout.addWidget(self.crop_btn)
    canvas_op_group.setLayout(canvas_op_layout)
    side_panel.addWidget(canvas_op_group)

    # 5. ファイル操作
    file_group = PySide6.QtWidgets.QGroupBox("ファイル")
    file_layout = PySide6.QtWidgets.QVBoxLayout()
    self.load_btn = PySide6.QtWidgets.QPushButton("画像を開く")
    self.load_btn.clicked.connect(self.load_file)
    self.save_btn = PySide6.QtWidgets.QPushButton("画像を保存")
    self.save_btn.clicked.connect(self.save_file)
    file_layout.addWidget(self.load_btn)
    file_layout.addWidget(self.save_btn)
    file_group.setLayout(file_layout)
    side_panel.addWidget(file_group)

    side_panel.addStretch()
    side_scroll.setWidget(side_container)
    main_layout.addWidget(side_scroll, 0)

    # --- 右側キャンバスエリア ---
    self.scroll_area = PySide6.QtWidgets.QScrollArea()
    self.canvas = PySide6.QtWidgets.QLabel()
    self.canvas.setAlignment(PySide6.QtCore.Qt.AlignmentFlag.AlignCenter)
    self.canvas.setStyleSheet("background-color: #222;")
    self.scroll_area.setWidget(self.canvas)
    self.scroll_area.setWidgetResizable(True)
    self.scroll_area.setAlignment(
        PySide6.QtCore.Qt.AlignmentFlag.AlignCenter)
    self.scroll_area.horizontalScrollBar().valueChanged.connect(
        lambda: self.update_navigator_frame())
    self.scroll_area.verticalScrollBar().valueChanged.connect(
        lambda: self.update_navigator_frame())
    main_layout.addWidget(self.scroll_area, 1)

    # ショートカット設定 (変更なし)
    self._setup_shortcuts()

  def _setup_shortcuts(self):
    PySide6.QtGui.QShortcut("Ctrl+Z", self).activated.connect(self.undo)
    PySide6.QtGui.QShortcut("Ctrl+Y", self).activated.connect(self.redo)
    PySide6.QtGui.QShortcut("B", self).activated.connect(
        lambda: self.mode_combo.setCurrentIndex(0))
    PySide6.QtGui.QShortcut("R", self).activated.connect(
        lambda: self.mode_combo.setCurrentIndex(1))
    PySide6.QtGui.QShortcut("C", self).activated.connect(
        lambda: self.mode_combo.setCurrentIndex(2))
    PySide6.QtGui.QShortcut("F", self).activated.connect(
        lambda: self.mode_combo.setCurrentIndex(3))
    PySide6.QtGui.QShortcut("E", self).activated.connect(
        self.eraser_btn.animateClick)
    PySide6.QtGui.QShortcut("P", self).activated.connect(
        self.brush_mode_btn.animateClick)
    PySide6.QtGui.QShortcut("[", self).activated.connect(
        lambda: self.brush_size_slider.setValue(self.brush_size_slider.value() - 2))
    PySide6.QtGui.QShortcut("]", self).activated.connect(
        lambda: self.brush_size_slider.setValue(self.brush_size_slider.value() + 2))
    PySide6.QtGui.QShortcut("Delete", self).activated.connect(
        self.delete_selection)
    PySide6.QtGui.QShortcut(
        "Ctrl+C", self).activated.connect(self.copy_selection)
    PySide6.QtGui.QShortcut(
        "Ctrl+V", self).activated.connect(self.paste_selection)
    PySide6.QtGui.QShortcut(
        "Ctrl+Shift+X", self).activated.connect(self.crop_to_selection)

  def _create_slider(self, label, min_v, max_v, init_v, layout):
    layout.addWidget(PySide6.QtWidgets.QLabel(label))
    slider = PySide6.QtWidgets.QSlider(
        PySide6.QtCore.Qt.Orientation.Horizontal)
    slider.setRange(min_v, max_v)
    slider.setValue(init_v)
    slider.valueChanged.connect(self.apply_effects)
    layout.addWidget(slider)
    return slider

  def add_color_to_history(self, color_bgr):
    if color_bgr in self.color_history: self.color_history.remove(color_bgr)
    self.color_history.insert(0, color_bgr)
    if len(self.color_history) > self.max_history: self.color_history.pop()
    self.update_history_ui()

  def update_history_ui(self):
    for i in reversed(range(self.history_grid.count())):
      self.history_grid.itemAt(i).widget().setParent(None)
    for i, color in enumerate(self.color_history):
      btn = PySide6.QtWidgets.QPushButton()
      btn.setFixedSize(25, 25)
      btn.setStyleSheet(
          f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); border: 1px solid #777;")
      btn.clicked.connect(lambda checked=False,
                          c=color: self.set_brush_color(c))
      self.history_grid.addWidget(btn, i // 4, i % 4)

  def set_brush_color(self, color_bgr):
    self.current_brush_color = color_bgr
    if self.eraser_mode:
      self.eraser_btn.setChecked(False)
      self.toggle_eraser()

  def change_draw_mode(self, index): self.draw_mode = index

  def toggle_brush_mode(self):
    self.pressure_brush_mode = self.brush_mode_btn.isChecked()
    self.brush_mode_btn.setText(
        "筆圧感度: ON" if self.pressure_brush_mode else "筆圧感度: OFF")

  def toggle_eraser(self):
    self.eraser_mode = self.eraser_btn.isChecked()
    self.eraser_btn.setText(f"消しゴム: {'ON' if self.eraser_mode else 'OFF'}")

  def create_blank_canvas(self, w, h):
    self.raw_image = np.full((h, w, 3), 255, dtype=np.uint8)
    self.undo_stack.clear(); self.redo_stack.clear(
    ); self.zoom_factor = 1.0; self.is_modified = False
    self.apply_effects()

  def resize_image_dialog(self):
    if self.raw_image is None: return
    h, w = self.raw_image.shape[:2]
    nw, ok1 = PySide6.QtWidgets.QInputDialog.getInt(
        self, "リサイズ", "幅:", w, 1, 10000)
    if not ok1: return
    nh, ok2 = PySide6.QtWidgets.QInputDialog.getInt(
        self, "リサイズ", "高さ:", h, 1, 10000)
    if not ok2: return
    self.save_undo_state()
    self.raw_image = cv2.resize(
        self.raw_image, (nw, nh), interpolation=cv2.INTER_AREA)
    self.apply_effects()

  def crop_to_selection(self):
    if self.raw_image is None or self.selection_rect is None: return
    r = self.selection_rect.normalized()
    ih, iw = self.raw_image.shape[:2]
    x1, y1, x2, y2 = max(0, r.left()), max(
        0, r.top()), min(iw, r.right()), min(ih, r.bottom())
    if x2 <= x1 or y2 <= y1: return
    self.save_undo_state()
    self.raw_image = self.raw_image[y1:y2, x1:x2].copy()
    self.selection_rect = None; self.apply_effects()

  def get_canvas_coordinates(self, pos: PySide6.QtCore.QPoint) -> typing.Optional[PySide6.QtCore.QPoint]:
    if self.raw_image is None or self.canvas.pixmap() is None: return None
    local_pos = self.canvas.mapFromGlobal(self.mapToGlobal(pos))
    pix_size = self.canvas.pixmap().size()
    offset_x, offset_y = (self.canvas.width() - pix_size.width()) / \
        2, (self.canvas.height() - pix_size.height()) / 2
    view_x, view_y = local_pos.x() - offset_x, local_pos.y() - offset_y
    ih, iw = self.raw_image.shape[:2]
    px, py = view_x * (iw / pix_size.width()), view_y * \
        (ih / pix_size.height())
    angle = self.rotate_slider.value()
    if angle != 0:
      M = cv2.getRotationMatrix2D((iw / 2, ih / 2), angle, 1.0)
      inv_M = cv2.invertAffineTransform(M)
      orig = inv_M @ np.array([px, py, 1.0])
      px, py = orig[0], orig[1]
    x, y = int(round(px)), int(round(py))
    if 0 <= x < iw and 0 <= y < ih: return PySide6.QtCore.QPoint(x, y)
    return None

  def wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None:
    if event.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.ControlModifier:
      pb = event.position()
      tp = self.get_canvas_coordinates(pb.toPoint())
      if event.angleDelta().y() > 0: self.zoom_factor *= 1.1
      else: self.zoom_factor /= 1.1
      self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
      self.apply_effects()
      if tp: self.adjust_scroll_to_pixel(tp, pb)
    else: super().wheelEvent(event)

  def adjust_scroll_to_pixel(self, pixel, screen_pos):
    ih, iw = self.raw_image.shape[:2]
    nvw, nvh = iw * self.zoom_factor, ih * self.zoom_factor
    tvx, tvy = pixel.x() * self.zoom_factor, pixel.y() * self.zoom_factor
    vpp = self.scroll_area.viewport().mapFromGlobal(
        self.mapToGlobal(screen_pos.toPoint()))
    mx, my = max(0, (self.canvas.width() - nvw) / 2), max(0,
                                                          (self.canvas.height() - nvh) / 2)
    self.scroll_area.horizontalScrollBar().setValue(int(tvx + mx - vpp.x()))
    self.scroll_area.verticalScrollBar().setValue(int(tvy + my - vpp.y()))

  def apply_effects(self, preview_pos=None) -> None:
    if self.raw_image is None: return
    img = self.raw_image.copy()
    if preview_pos and self.start_point:
      color = self.current_bg_color if self.eraser_mode else self.current_brush_color
      thick = self.brush_size_slider.value()
      if self.draw_mode == 0 and PySide6.QtWidgets.QApplication.keyboardModifiers() & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier:
        cv2.line(img, (self.start_point.x(), self.start_point.y()),
                 (preview_pos.x(), preview_pos.y()), color, thick)
      elif self.draw_mode == 1: cv2.rectangle(img, (self.start_point.x(), self.start_point.y()), (preview_pos.x(), preview_pos.y()), color, thick)
      elif self.draw_mode == 2:
        c = (self.start_point.x(), self.start_point.y())
        r = int(np.linalg.norm(
            [preview_pos.x() - c[0], preview_pos.y() - c[1]]))
        cv2.circle(img, c, r, color, thick)
    eff = self.effect_combo.currentIndex()
    if eff == 1: img = cv2.cvtColor(cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif eff == 2: img = cv2.transform(img, np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
    elif eff == 3:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      edge = cv2.adaptiveThreshold(
          gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
      img = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    k = self.blur_slider.value()
    if k > 1: img = cv2.GaussianBlur(img, (k | 1, k | 1), 0)
    angle = self.rotate_slider.value()
    if angle != 0:
      h, w = img.shape[:2]; M = cv2.getRotationMatrix2D(
          (w / 2, h / 2), angle, 1.0)
      img = cv2.warpAffine(
          img, M, (w, h), borderValue=self.current_bg_color)
    csel = None
    if self.draw_mode == 4 and self.start_point and preview_pos: csel = PySide6.QtCore.QRect(
        self.start_point, preview_pos).normalized()
    elif self.selection_rect: csel = self.selection_rect
    if csel:
      h, w = self.raw_image.shape[:2]; M = cv2.getRotationMatrix2D(
          (w / 2, h / 2), angle, 1.0)
      pts = np.array([[csel.left(), csel.top()], [csel.right(), csel.top()], [
                     csel.right(), csel.bottom()], [csel.left(), csel.bottom()]], dtype='float32')
      pts_r = M.dot(np.concatenate(
          [pts, np.ones((4, 1))], axis=1).T).T.astype(int)
      cv2.polylines(img, [pts_r], True, (0, 0, 0), 2); cv2.polylines(
          img, [pts_r], True, (255, 255, 255), 1)
    self.display_image(img)

  def display_image(self, img):
    h, w, ch = img.shape
    q_img = PySide6.QtGui.QImage(
        img.data, w, h, ch * w, PySide6.QtGui.QImage.Format.Format_BGR888)
    pix = PySide6.QtGui.QPixmap.fromImage(q_img)
    self.nav_pixmap = pix.scaled(self.nav_label.size(
    ), PySide6.QtCore.Qt.AspectRatioMode.KeepAspectRatio, PySide6.QtCore.Qt.TransformationMode.SmoothTransformation)
    self.update_navigator_frame()
    spix = pix.scaled(pix.size() * self.zoom_factor, PySide6.QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                      PySide6.QtCore.Qt.TransformationMode.SmoothTransformation)
    self.canvas.setPixmap(spix); self.canvas.setFixedSize(spix.size())

  def update_navigator_frame(self):
    if not hasattr(self, 'nav_pixmap') or self.nav_pixmap.isNull(): return
    rnav = self.nav_pixmap.copy(); painter = PySide6.QtGui.QPainter(rnav)
    vb, hb = self.scroll_area.verticalScrollBar(), self.scroll_area.horizontalScrollBar()
    tw, th = self.canvas.width(), self.canvas.height()
    vw, vh = self.scroll_area.viewport().width(), self.scroll_area.viewport().height()
    if tw > 0 and th > 0:
      rx, ry = (hb.value() / tw) * \
          rnav.width(), (vb.value() / th) * rnav.height()
      rw, rh = (vw / tw) * rnav.width(), (vh / th) * rnav.height()
      painter.setPen(PySide6.QtGui.QPen(
          PySide6.QtGui.QColor(255, 0, 0), 2))
      painter.drawRect(PySide6.QtCore.QRectF(rx, ry, rw, rh))
    painter.end(); self.nav_label.setPixmap(rnav)

  def navigator_mouse_event(self, event):
    if self.raw_image is None: return
    if event.buttons() & PySide6.QtCore.Qt.MouseButton.LeftButton:
      npax = self.nav_label.pixmap().size()
      ox, oy = (self.nav_label.width() - npax.width()) / \
          2, (self.nav_label.height() - npax.height()) / 2
      rx, ry = (event.position().x() - ox) / \
          npax.width(), (event.position().y() - oy) / npax.height()
      self.scroll_area.horizontalScrollBar().setValue(
          int(rx * self.canvas.width() - self.scroll_area.viewport().width() / 2))
      self.scroll_area.verticalScrollBar().setValue(
          int(ry * self.canvas.height() - self.scroll_area.viewport().height() / 2))

  def keyPressEvent(self, e):
    if e.key() == PySide6.QtCore.Qt.Key.Key_Space and not e.isAutoRepeat():
      self.is_panning = True; self.canvas.setCursor(
          PySide6.QtCore.Qt.CursorShape.ClosedHandCursor)
    super().keyPressEvent(e)

  def keyReleaseEvent(self, e):
    if e.key() == PySide6.QtCore.Qt.Key.Key_Space and not e.isAutoRepeat():
      self.is_panning = False; self.canvas.setCursor(
          PySide6.QtCore.Qt.CursorShape.ArrowCursor)
    super().keyReleaseEvent(e)

  def mousePressEvent(self, e):
    if self.is_panning: self.last_mouse_pos = e.pos(); return
    p = self.get_canvas_coordinates(e.pos())
    if not p: self.selection_rect = None; self.apply_effects(); return
    if e.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.AltModifier:
      bgr = self.raw_image[p.y(), p.x()]; color = (
          int(bgr[0]), int(bgr[1]), int(bgr[2]))
      self.current_brush_color = color; self.add_color_to_history(color); return
    if self.draw_mode != 4: self.selection_rect = None
    self.save_undo_state()
    if self.draw_mode == 3:
      cv2.floodFill(self.raw_image, None, (p.x(), p.y(
      )), self.current_bg_color if self.eraser_mode else self.current_brush_color)
      self.apply_effects()
    else: self.start_point = self.last_point = p; self.last_time = time.time()

  def mouseMoveEvent(self, e):
    if self.is_panning:
      d = e.pos() - self.last_mouse_pos; self.last_mouse_pos = e.pos()
      self.scroll_area.horizontalScrollBar().setValue(
          self.scroll_area.horizontalScrollBar().value() - d.x())
      self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() - d.y()); return
    if e.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.AltModifier or not (e.buttons() & PySide6.QtCore.Qt.MouseButton.LeftButton) or not self.start_point: return
    p = self.get_canvas_coordinates(e.pos())
    if not p: return
    if self.draw_mode == 0:
      if e.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier: self.apply_effects(preview_pos=p)
      else:
        c = self.current_bg_color if self.eraser_mode else self.current_brush_color
        t = self.brush_size_slider.value()
        if self.pressure_brush_mode:
          v = np.linalg.norm([p.x() - self.last_point.x(), p.y() - self.last_point.y()]) / (
              time.time() - self.last_time + 0.001)
          self.current_velocity_size = self.current_velocity_size * \
              0.8 + (t * max(0.2, min(1.2, 80 / (v + 1)))) * 0.2
          t, self.last_time = int(
              self.current_velocity_size), time.time()
        cv2.line(self.raw_image, (self.last_point.x(),
                 self.last_point.y()), (p.x(), p.y()), c, max(1, t))
        self.last_point = p; self.apply_effects()
    else: self.apply_effects(preview_pos=p)

  def mouseReleaseEvent(self, e):
    if self.is_panning: return
    if self.start_point:
      p = self.get_canvas_coordinates(e.pos())
      if p:
        c = self.current_bg_color if self.eraser_mode else self.current_brush_color
        t = self.brush_size_slider.value()
        if self.draw_mode == 0 and e.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier: cv2.line(
            self.raw_image, (self.start_point.x(), self.start_point.y()), (p.x(), p.y()), c, t)
        elif self.draw_mode == 1: cv2.rectangle(self.raw_image, (self.start_point.x(), self.start_point.y()), (p.x(), p.y()), c, t)
        elif self.draw_mode == 2:
          cp = (self.start_point.x(), self.start_point.y())
          cv2.circle(self.raw_image, cp, int(
              np.linalg.norm([p.x() - cp[0], p.y() - cp[1]])), c, t)
        elif self.draw_mode == 4: self.selection_rect = PySide6.QtCore.QRect(self.start_point, p).normalized()
        self.apply_effects()
    self.start_point = self.last_point = None

  def delete_selection(self):
    if self.selection_rect and self.raw_image is not None:
      self.save_undo_state(); r = self.selection_rect
      cv2.rectangle(self.raw_image, (r.left(), r.top()), (r.right(
      ), r.bottom()), self.current_bg_color, -1); self.apply_effects()

  def copy_selection(self):
    if self.selection_rect and self.raw_image is not None:
      r = self.selection_rect; self.clipboard = self.raw_image[r.top(
      ):r.bottom(), r.left():r.right()].copy()

  def paste_selection(self):
    if self.clipboard is not None and self.raw_image is not None:
      self.save_undo_state(
      ); h, w = self.clipboard.shape[:2]; t = self.last_point if self.last_point else PySide6.QtCore.QPoint(0, 0)
      x2, y2 = min(self.raw_image.shape[1], t.x(
      ) + w), min(self.raw_image.shape[0], t.y() + h)
      self.raw_image[t.y():y2, t.x():x2] = self.clipboard[:y2 -
                                                          t.y(), :x2 - t.x()]
      self.selection_rect = PySide6.QtCore.QRect(
          t.x(), t.y(), x2 - t.x(), y2 - t.y()); self.apply_effects()

  def save_undo_state(self):
    self.undo_stack.append(self.raw_image.copy())
    if len(self.undo_stack) > self.max_undo: self.undo_stack.pop(0)
    self.redo_stack.clear(); self.is_modified = True

  def undo(self):
    if self.undo_stack: self.redo_stack.append(self.raw_image.copy(
    )); self.raw_image = self.undo_stack.pop(); self.apply_effects()

  def redo(self):
    if self.redo_stack: self.undo_stack.append(self.raw_image.copy(
    )); self.raw_image = self.redo_stack.pop(); self.apply_effects()

  def change_brush_color(self):
    c = PySide6.QtWidgets.QColorDialog.getColor()
    if c.isValid(): color = (c.blue(), c.green(), c.red()); self.set_brush_color(
        color); self.add_color_to_history(color)

  def clear_canvas_with_undo(self):
    self.save_undo_state(); self.raw_image[:] = 255; self.apply_effects()

  def load_file(self):
    path, _ = PySide6.QtWidgets.QFileDialog.getOpenFileName(
        self, "画像を開く", "", "Images (*.png *.jpg)")
    if path: self.save_undo_state(); self.raw_image = cv2.imdecode(np.fromfile(path, np.uint8),
                                                                   cv2.IMREAD_COLOR); self.zoom_factor = 1.0; self.is_modified = False; self.apply_effects()

  def save_file(self):
    path, _ = PySide6.QtWidgets.QFileDialog.getSaveFileName(
        self, "画像を保存", "output.png", "PNG (*.png)")
    if path: cv2.imencode(".png", self.raw_image)[1].tofile(path); self.is_modified = False; return True
    return False

  def closeEvent(self, event):
    if not self.is_modified: event.accept(); return
    reply = PySide6.QtWidgets.QMessageBox.question(self, "確認", "保存しますか？", PySide6.QtWidgets.QMessageBox.StandardButton.Save |
                                                   PySide6.QtWidgets.QMessageBox.StandardButton.Discard | PySide6.QtWidgets.QMessageBox.StandardButton.Cancel)
    if reply == PySide6.QtWidgets.QMessageBox.StandardButton.Save:
      if self.save_file(): event.accept()
      else: event.ignore()
    elif reply == PySide6.QtWidgets.QMessageBox.StandardButton.Discard: event.accept()
    else: event.ignore()

if __name__ == "__main__":
  app = PySide6.QtWidgets.QApplication(sys.argv)
  window = AdvancedImageApp()
  window.show()
  sys.exit(app.exec())
