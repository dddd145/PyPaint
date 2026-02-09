import sys
import cv2
import numpy as np
import typing
import time
import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui

# メインアプリケーションクラス：PySide6のQMainWindowを継承
class AdvancedImageApp(PySide6.QtWidgets.QMainWindow):
  def __init__(self) -> None:
    super().__init__()
    self.setWindowTitle("PyPainter")
    self.setMinimumSize(1200, 800)  # ナビゲーター用に少し広めに設定

    # --- 状態保持用の変数 ---
    self.raw_image: typing.Optional[np.ndarray] = None
    self.last_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.start_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.eraser_mode = False
    self.pressure_brush_mode = False
    self.current_bg_color = (255, 255, 255)
    self.current_brush_color = (0, 0, 0)
    self.zoom_factor = 1.0

    self.draw_mode = 0  # 0:ブラシ, 1:矩形, 2:円, 3:塗りつぶし, 4:選択
    self.selection_rect: typing.Optional[PySide6.QtCore.QRect] = None
    self.clipboard: typing.Optional[np.ndarray] = None
    self.is_modified = False
    self.last_time = 0.0
    self.current_velocity_size = 5.0

    self.is_panning = False
    self.last_mouse_pos = PySide6.QtCore.QPoint()

    # Undo/Redo管理
    self.undo_stack: typing.List[np.ndarray] = []
    self.redo_stack: typing.List[np.ndarray] = []
    self.max_undo = 30

    self.init_ui()
    self.create_blank_canvas(800, 600)

  def init_ui(self) -> None:
    central_widget = PySide6.QtWidgets.QWidget()
    self.setCentralWidget(central_widget)
    main_layout = PySide6.QtWidgets.QHBoxLayout(central_widget)

    # --- 左側サイドパネル ---
    side_panel = PySide6.QtWidgets.QVBoxLayout()

    # --- [NEW] ナビゲーターエリア ---
    nav_group = PySide6.QtWidgets.QGroupBox("ナビゲーター")
    nav_layout = PySide6.QtWidgets.QVBoxLayout()
    self.nav_label = PySide6.QtWidgets.QLabel()
    self.nav_label.setFixedSize(200, 150)
    self.nav_label.setAlignment(PySide6.QtCore.Qt.AlignmentFlag.AlignCenter)
    self.nav_label.setStyleSheet(
        "background-color: #000; border: 1px solid #555;")
    # ナビゲーター上でのクリックイベントを拾えるようにする
    self.nav_label.mousePressEvent = self.navigator_mouse_event
    self.nav_label.mouseMoveEvent = self.navigator_mouse_event
    nav_layout.addWidget(self.nav_label)
    nav_group.setLayout(nav_layout)
    side_panel.addWidget(nav_group)

    draw_group = PySide6.QtWidgets.QGroupBox("ツール設定")
    draw_layout = PySide6.QtWidgets.QVBoxLayout()

    self.brush_size_slider = self._create_slider(
        "サイズ", 1, 50, 5, draw_layout)

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

    self.brush_color_btn = PySide6.QtWidgets.QPushButton("色を変える")
    self.brush_color_btn.clicked.connect(self.change_brush_color)
    draw_layout.addWidget(self.brush_color_btn)

    self.eraser_btn = PySide6.QtWidgets.QPushButton("消しゴム: OFF")
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

    self.clear_btn = PySide6.QtWidgets.QPushButton("クリア")
    self.clear_btn.clicked.connect(self.clear_canvas_with_undo)
    draw_layout.addWidget(self.clear_btn)

    draw_group.setLayout(draw_layout)
    side_panel.addWidget(draw_group)

    # --- エフェクト設定パネル ---
    proc_group = PySide6.QtWidgets.QGroupBox("エフェクト")
    proc_layout = PySide6.QtWidgets.QVBoxLayout()

    self.rotate_slider = self._create_slider(
        "回転", -180, 180, 0, proc_layout)
    self.blur_slider = self._create_slider("ぼかし", 1, 51, 1, proc_layout)

    proc_layout.addWidget(PySide6.QtWidgets.QLabel("特殊効果:"))
    self.effect_combo = PySide6.QtWidgets.QComboBox()
    self.effect_combo.addItems(["なし", "モノクロ", "セピア", "エッジ抽出"])
    self.effect_combo.currentIndexChanged.connect(self.apply_effects)
    proc_layout.addWidget(self.effect_combo)

    proc_group.setLayout(proc_layout)
    side_panel.addWidget(proc_group)

    # --- [NEW] キャンバス操作パネル ---
    canvas_op_group = PySide6.QtWidgets.QGroupBox("キャンバス操作")
    canvas_op_layout = PySide6.QtWidgets.QVBoxLayout()

    self.resize_btn = PySide6.QtWidgets.QPushButton("リサイズ (画像解像度)")
    self.resize_btn.clicked.connect(self.resize_image_dialog)
    canvas_op_layout.addWidget(self.resize_btn)

    self.crop_btn = PySide6.QtWidgets.QPushButton("選択範囲でトリミング")
    self.crop_btn.clicked.connect(self.crop_to_selection)
    canvas_op_layout.addWidget(self.crop_btn)

    canvas_op_group.setLayout(canvas_op_layout)
    side_panel.addWidget(canvas_op_group)

    self.load_btn = PySide6.QtWidgets.QPushButton("読み込み")
    self.load_btn.clicked.connect(self.load_file)
    side_panel.addWidget(self.load_btn)
    self.save_btn = PySide6.QtWidgets.QPushButton("保存")
    self.save_btn.clicked.connect(self.save_file)
    side_panel.addWidget(self.save_btn)

    side_panel.addStretch()
    main_layout.addLayout(side_panel, 1)

    # --- 右側キャンバスエリア ---
    self.scroll_area = PySide6.QtWidgets.QScrollArea()
    self.canvas = PySide6.QtWidgets.QLabel()
    self.canvas.setAlignment(PySide6.QtCore.Qt.AlignmentFlag.AlignCenter)
    self.canvas.setStyleSheet("background-color: #222;")
    self.scroll_area.setWidget(self.canvas)
    self.scroll_area.setWidgetResizable(True)
    self.scroll_area.setAlignment(
        PySide6.QtCore.Qt.AlignmentFlag.AlignCenter)
    # スクロールが発生した際にナビゲーターを更新
    self.scroll_area.horizontalScrollBar().valueChanged.connect(
        lambda: self.update_navigator_frame())
    self.scroll_area.verticalScrollBar().valueChanged.connect(
        lambda: self.update_navigator_frame())
    main_layout.addWidget(self.scroll_area, 4)

    # --- ショートカットキー ---
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
    self.undo_stack.clear()
    self.redo_stack.clear()
    self.zoom_factor = 1.0
    self.is_modified = False
    self.apply_effects()

  # --- [NEW] リサイズ・トリミング機能の実装 ---
  def resize_image_dialog(self):
    if self.raw_image is None: return
    h, w = self.raw_image.shape[:2]

    # 幅の入力
    new_w, ok1 = PySide6.QtWidgets.QInputDialog.getInt(
        self, "リサイズ", "新しい幅 (px):", w, 1, 10000)
    if not ok1: return

    # 高の入力
    new_h, ok2 = PySide6.QtWidgets.QInputDialog.getInt(
        self, "リサイズ", "新しい高さ (px):", h, 1, 10000)
    if not ok2: return

    self.save_undo_state()
    self.raw_image = cv2.resize(
        self.raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    self.apply_effects()

  def crop_to_selection(self):
    if self.raw_image is None or self.selection_rect is None:
      PySide6.QtWidgets.QMessageBox.information(
          self, "情報", "トリミングする範囲を選択してください。")
      return

    r = self.selection_rect.normalized()
    img_h, img_w = self.raw_image.shape[:2]

    # 範囲が画像内にあるかクリッピングして確認
    x1, y1 = max(0, r.left()), max(0, r.top())
    x2, y2 = min(img_w, r.right()), min(img_h, r.bottom())

    if x2 <= x1 or y2 <= y1: return

    self.save_undo_state()
    self.raw_image = self.raw_image[y1:y2, x1:x2].copy()
    self.selection_rect = None
    self.apply_effects()

  def get_canvas_coordinates(self, pos: PySide6.QtCore.QPoint) -> typing.Optional[PySide6.QtCore.QPoint]:
    if self.raw_image is None or self.canvas.pixmap() is None: return None
    local_pos = self.canvas.mapFromGlobal(self.mapToGlobal(pos))
    pix_size = self.canvas.pixmap().size()
    offset_x = (self.canvas.width() - pix_size.width()) / 2
    offset_y = (self.canvas.height() - pix_size.height()) / 2
    view_x = local_pos.x() - offset_x
    view_y = local_pos.y() - offset_y
    img_h, img_w = self.raw_image.shape[:2]
    px = view_x * (img_w / pix_size.width())
    py = view_y * (img_h / pix_size.height())
    angle = self.rotate_slider.value()
    if angle != 0:
      matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
      inv_matrix = cv2.invertAffineTransform(matrix)
      original_point = inv_matrix @ np.array([px, py, 1.0])
      px, py = original_point[0], original_point[1]
    x, y = int(round(px)), int(round(py))
    if 0 <= x < img_w and 0 <= y < img_h: return PySide6.QtCore.QPoint(x, y)
    return None

  def wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None:
    if event.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.ControlModifier:
      pos_before = event.position()
      target_pixel = self.get_canvas_coordinates(pos_before.toPoint())
      if event.angleDelta().y() > 0: self.zoom_factor *= 1.1
      else: self.zoom_factor /= 1.1
      self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
      self.apply_effects()
      if target_pixel: self.adjust_scroll_to_pixel(
          target_pixel, pos_before)
    else: super().wheelEvent(event)

  def adjust_scroll_to_pixel(self, pixel, screen_pos):
    img_h, img_w = self.raw_image.shape[:2]
    new_view_w, new_view_h = img_w * self.zoom_factor, img_h * self.zoom_factor
    target_view_x, target_view_y = pixel.x() * self.zoom_factor, pixel.y() * \
        self.zoom_factor
    viewport_pos = self.scroll_area.viewport().mapFromGlobal(
        self.mapToGlobal(screen_pos.toPoint()))
    margin_x = max(0, (self.canvas.width() - new_view_w) / 2)
    margin_y = max(0, (self.canvas.height() - new_view_h) / 2)
    self.scroll_area.horizontalScrollBar().setValue(
        int(target_view_x + margin_x - viewport_pos.x()))
    self.scroll_area.verticalScrollBar().setValue(
        int(target_view_y + margin_y - viewport_pos.y()))

  def apply_effects(self, preview_pos=None) -> None:
    if self.raw_image is None: return
    img = self.raw_image.copy()

    if preview_pos and self.start_point:
      color = self.current_bg_color if self.eraser_mode else self.current_brush_color
      thickness = self.brush_size_slider.value()
      modifiers = PySide6.QtWidgets.QApplication.keyboardModifiers()
      if self.draw_mode == 0 and modifiers & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier:
        cv2.line(img, (self.start_point.x(), self.start_point.y()),
                 (preview_pos.x(), preview_pos.y()), color, thickness)
      elif self.draw_mode == 1:
        cv2.rectangle(img, (self.start_point.x(), self.start_point.y(
        )), (preview_pos.x(), preview_pos.y()), color, thickness)
      elif self.draw_mode == 2:
        center = (self.start_point.x(), self.start_point.y())
        radius = int(np.linalg.norm(
            [preview_pos.x() - center[0], preview_pos.y() - center[1]]))
        cv2.circle(img, center, radius, color, thickness)

    eff = self.effect_combo.currentIndex()
    if eff == 1: img = cv2.cvtColor(cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif eff == 2:
      kernel = np.array(
          [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
      img = cv2.transform(img, kernel)
    elif eff == 3:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      edge = cv2.adaptiveThreshold(
          gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
      img = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    k = self.blur_slider.value()
    if k > 1: img = cv2.GaussianBlur(img, (k | 1, k | 1), 0)

    angle = self.rotate_slider.value()
    if angle != 0:
      h, w = img.shape[:2]
      M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
      img = cv2.warpAffine(
          img, M, (w, h), borderValue=self.current_bg_color)

    current_sel = None
    if self.draw_mode == 4 and self.start_point and preview_pos:
      current_sel = PySide6.QtCore.QRect(
          self.start_point, preview_pos).normalized()
    elif self.selection_rect:
      current_sel = self.selection_rect

    if current_sel:
      h, w = self.raw_image.shape[:2]
      M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
      pts = np.array([[current_sel.left(), current_sel.top()], [current_sel.right(), current_sel.top()],
                      [current_sel.right(), current_sel.bottom()], [current_sel.left(), current_sel.bottom()]], dtype='float32')
      ones = np.ones(shape=(len(pts), 1))
      pts_ones = np.concatenate([pts, ones], axis=1)
      rotated_pts = M.dot(pts_ones.T).T.astype(int)
      cv2.polylines(img, [rotated_pts], True, (0, 0, 0), 2)
      cv2.polylines(img, [rotated_pts], True,
                    (255, 255, 255), 1, cv2.LINE_4)

    self.display_image(img)

  def display_image(self, img):
    h, w, ch = img.shape
    q_img = PySide6.QtGui.QImage(
        img.data, w, h, ch * w, PySide6.QtGui.QImage.Format.Format_BGR888)
    pix = PySide6.QtGui.QPixmap.fromImage(q_img)

    self.nav_pixmap = pix.scaled(self.nav_label.size(),
                                 PySide6.QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                 PySide6.QtCore.Qt.TransformationMode.SmoothTransformation)
    self.update_navigator_frame()

    scaled_pix = pix.scaled(pix.size() * self.zoom_factor,
                            PySide6.QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            PySide6.QtCore.Qt.TransformationMode.SmoothTransformation)
    self.canvas.setPixmap(scaled_pix)
    self.canvas.setFixedSize(scaled_pix.size())

  def update_navigator_frame(self):
    if not hasattr(self, 'nav_pixmap') or self.nav_pixmap.isNull(): return
    result_nav = self.nav_pixmap.copy()
    painter = PySide6.QtGui.QPainter(result_nav)
    v_bar = self.scroll_area.verticalScrollBar()
    h_bar = self.scroll_area.horizontalScrollBar()
    total_w = self.canvas.width()
    total_h = self.canvas.height()
    view_w = self.scroll_area.viewport().width()
    view_h = self.scroll_area.viewport().height()

    if total_w > 0 and total_h > 0:
      rect_x = (h_bar.value() / total_w) * result_nav.width()
      rect_y = (v_bar.value() / total_h) * result_nav.height()
      rect_w = (view_w / total_w) * result_nav.width()
      rect_h = (view_h / total_h) * result_nav.height()
      painter.setPen(PySide6.QtGui.QPen(
          PySide6.QtGui.QColor(255, 0, 0), 2))
      painter.drawRect(PySide6.QtCore.QRectF(
          rect_x, rect_y, rect_w, rect_h))
    painter.end()
    self.nav_label.setPixmap(result_nav)

  def navigator_mouse_event(self, event):
    if self.raw_image is None: return
    if event.buttons() & PySide6.QtCore.Qt.MouseButton.LeftButton:
      nav_pos = event.position()
      pix_size = self.nav_label.pixmap().size()
      off_x = (self.nav_label.width() - pix_size.width()) / 2
      off_y = (self.nav_label.height() - pix_size.height()) / 2
      rel_x = (nav_pos.x() - off_x) / pix_size.width()
      rel_y = (nav_pos.y() - off_y) / pix_size.height()
      h_bar = self.scroll_area.horizontalScrollBar()
      v_bar = self.scroll_area.verticalScrollBar()
      target_h = rel_x * self.canvas.width() - self.scroll_area.viewport().width() / 2
      target_v = rel_y * self.canvas.height() - self.scroll_area.viewport().height() / 2
      h_bar.setValue(int(target_h))
      v_bar.setValue(int(target_v))

  def keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
    if event.key() == PySide6.QtCore.Qt.Key.Key_Space:
      if not event.isAutoRepeat():
        self.is_panning = True
        self.canvas.setCursor(
            PySide6.QtCore.Qt.CursorShape.ClosedHandCursor)
    super().keyPressEvent(event)

  def keyReleaseEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
    if event.key() == PySide6.QtCore.Qt.Key.Key_Space:
      if not event.isAutoRepeat():
        self.is_panning = False
        self.canvas.setCursor(PySide6.QtCore.Qt.CursorShape.ArrowCursor)
    super().keyReleaseEvent(event)

  def mousePressEvent(self, event):
    if self.is_panning:
      self.last_mouse_pos = event.pos()
      return
    pos = self.get_canvas_coordinates(event.pos())
    if not pos:
      self.selection_rect = None
      self.apply_effects()
      return
    if event.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.AltModifier:
      bgr = self.raw_image[pos.y(), pos.x()]
      self.current_brush_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
      return
    if self.draw_mode != 4:
      self.selection_rect = None
    self.save_undo_state()
    if self.draw_mode == 3:
      color = self.current_bg_color if self.eraser_mode else self.current_brush_color
      cv2.floodFill(self.raw_image, None, (pos.x(), pos.y()), color)
      self.apply_effects()
    else:
      self.start_point = self.last_point = pos
      self.last_time = time.time()

  def mouseMoveEvent(self, event):
    if self.is_panning:
      delta = event.pos() - self.last_mouse_pos
      self.last_mouse_pos = event.pos()
      h_bar = self.scroll_area.horizontalScrollBar()
      v_bar = self.scroll_area.verticalScrollBar()
      h_bar.setValue(h_bar.value() - delta.x())
      v_bar.setValue(v_bar.value() - delta.y())
      return
    if event.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.AltModifier:
      return
    if not (event.buttons() & PySide6.QtCore.Qt.MouseButton.LeftButton) or not self.start_point: return
    pos = self.get_canvas_coordinates(event.pos())
    if not pos: return
    if self.draw_mode == 0:
      modifiers = event.modifiers()
      if modifiers & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier:
        self.apply_effects(preview_pos=pos)
      else:
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color
        thick = self.brush_size_slider.value()
        if self.pressure_brush_mode:
          now = time.time()
          dist = np.linalg.norm(
              [pos.x() - self.last_point.x(), pos.y() - self.last_point.y()])
          v = dist / (now - self.last_time + 0.001)
          self.current_velocity_size = self.current_velocity_size * \
              0.8 + (thick * max(0.2, min(1.2, 80 / (v + 1)))) * 0.2
          thick = int(self.current_velocity_size)
          self.last_time = now
        cv2.line(self.raw_image, (self.last_point.x(), self.last_point.y(
        )), (pos.x(), pos.y()), color, max(1, thick))
        self.last_point = pos
        self.apply_effects()
    else:
      self.apply_effects(preview_pos=pos)

  def mouseReleaseEvent(self, event):
    if self.is_panning: return
    if self.start_point:
      pos = self.get_canvas_coordinates(event.pos())
      if pos:
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color
        thick = self.brush_size_slider.value()
        modifiers = event.modifiers()
        if self.draw_mode == 0 and modifiers & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier:
          cv2.line(self.raw_image, (self.start_point.x(
          ), self.start_point.y()), (pos.x(), pos.y()), color, thick)
        elif self.draw_mode == 1:
          cv2.rectangle(self.raw_image, (self.start_point.x(
          ), self.start_point.y()), (pos.x(), pos.y()), color, thick)
        elif self.draw_mode == 2:
          center = (self.start_point.x(), self.start_point.y())
          radius = int(np.linalg.norm(
              [pos.x() - center[0], pos.y() - center[1]]))
          cv2.circle(self.raw_image, center, radius, color, thick)
        elif self.draw_mode == 4:
          self.selection_rect = PySide6.QtCore.QRect(
              self.start_point, pos).normalized()
        self.apply_effects()
    self.start_point = self.last_point = None

  def delete_selection(self):
    if self.selection_rect and self.raw_image is not None:
      self.save_undo_state()
      r = self.selection_rect
      cv2.rectangle(self.raw_image, (r.left(), r.top()),
                    (r.right(), r.bottom()), self.current_bg_color, -1)
      self.apply_effects()

  def copy_selection(self):
    if self.selection_rect and self.raw_image is not None:
      r = self.selection_rect
      self.clipboard = self.raw_image[r.top(
      ):r.bottom(), r.left():r.right()].copy()

  def paste_selection(self):
    if self.clipboard is not None and self.raw_image is not None:
      self.save_undo_state()
      h, w = self.clipboard.shape[:2]
      target = self.last_point if self.last_point else PySide6.QtCore.QPoint(
          0, 0)
      x2, y2 = min(self.raw_image.shape[1], target.x(
      ) + w), min(self.raw_image.shape[0], target.y() + h)
      self.raw_image[target.y():y2, target.x(
      ):x2] = self.clipboard[:y2 - target.y(), :x2 - target.x()]
      self.selection_rect = PySide6.QtCore.QRect(
          target.x(), target.y(), x2 - target.x(), y2 - target.y())
      self.apply_effects()

  def save_undo_state(self):
    self.undo_stack.append(self.raw_image.copy())
    if len(self.undo_stack) > self.max_undo: self.undo_stack.pop(0)
    self.redo_stack.clear()
    self.is_modified = True

  def undo(self):
    if self.undo_stack:
      self.redo_stack.append(self.raw_image.copy())
      self.raw_image = self.undo_stack.pop()
      self.apply_effects()

  def redo(self):
    if self.redo_stack:
      self.undo_stack.append(self.raw_image.copy())
      self.raw_image = self.redo_stack.pop()
      self.apply_effects()

  def change_brush_color(self):
    c = PySide6.QtWidgets.QColorDialog.getColor()
    if c.isValid(): self.current_brush_color = (c.blue(), c.green(), c.red())

  def clear_canvas_with_undo(self):
    self.save_undo_state()
    self.raw_image[:] = 255
    self.apply_effects()

  def load_file(self):
    path, _ = PySide6.QtWidgets.QFileDialog.getOpenFileName(
        self, "画像を開く", "", "Images (*.png *.jpg)")
    if path:
      self.save_undo_state()
      self.raw_image = cv2.imdecode(
          np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
      self.zoom_factor = 1.0
      self.is_modified = False
      self.apply_effects()

  def save_file(self):
    path, _ = PySide6.QtWidgets.QFileDialog.getSaveFileName(
        self, "画像を保存", "output.png", "PNG (*.png)")
    if path:
      cv2.imencode(".png", self.raw_image)[1].tofile(path)
      self.is_modified = False
      return True
    return False

  def closeEvent(self, event: PySide6.QtGui.QCloseEvent):
    if not self.is_modified:
      event.accept()
      return
    reply = PySide6.QtWidgets.QMessageBox.question(self, "確認", "変更が保存されていません。終了する前に保存しますか？",
                                                   PySide6.QtWidgets.QMessageBox.StandardButton.Save |
                                                   PySide6.QtWidgets.QMessageBox.StandardButton.Discard |
                                                   PySide6.QtWidgets.QMessageBox.StandardButton.Cancel,
                                                   PySide6.QtWidgets.QMessageBox.StandardButton.Save)
    if reply == PySide6.QtWidgets.QMessageBox.StandardButton.Save:
      if self.save_file(): event.accept()
      else: event.ignore()
    elif reply == PySide6.QtWidgets.QMessageBox.StandardButton.Discard:
      event.accept()
    else:
      event.ignore()

if __name__ == "__main__":
  app = PySide6.QtWidgets.QApplication(sys.argv)
  window = AdvancedImageApp()
  window.show()
  sys.exit(app.exec())
