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
    self.setWindowTitle("PyPainter Pro")
    self.setMinimumSize(1000, 700)

    # --- 状態保持用の変数 ---
    self.raw_image: typing.Optional[np.ndarray] = None
    self.last_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.start_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.eraser_mode = False
    self.pressure_brush_mode = False
    self.current_bg_color = (255, 255, 255)
    self.current_brush_color = (0, 0, 0)
    self.zoom_factor = 1.0

    self.draw_mode = 0
    self.is_modified = False
    self.last_time = 0.0
    self.current_velocity_size = 5.0

    self.undo_stack: typing.List[np.ndarray] = []
    self.redo_stack: typing.List[np.ndarray] = []
    self.max_undo = 30

    self.init_ui()
    self.create_blank_canvas(800, 600)

  def init_ui(self) -> None:
    central_widget = PySide6.QtWidgets.QWidget()
    self.setCentralWidget(central_widget)
    main_layout = PySide6.QtWidgets.QHBoxLayout(central_widget)

    side_panel = PySide6.QtWidgets.QVBoxLayout()

    draw_group = PySide6.QtWidgets.QGroupBox("ツール設定")
    draw_layout = PySide6.QtWidgets.QVBoxLayout()
    self.brush_size_slider = self._create_slider(
        "サイズ", 1, 50, 5, draw_layout)
    self.mode_combo = PySide6.QtWidgets.QComboBox()
    self.mode_combo.addItems(
        ["通常ブラシ", "矩形 (Rect)", "円 (Circle)", "塗りつぶし (Fill)"])
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

    proc_group = PySide6.QtWidgets.QGroupBox("エフェクト")
    proc_layout = PySide6.QtWidgets.QVBoxLayout()
    self.rotate_slider = self._create_slider(
        "回転", -180, 180, 0, proc_layout)
    self.blur_slider = self._create_slider("ぼかし", 1, 51, 1, proc_layout)
    self.effect_slider = self._create_slider("特殊効果", 0, 3, 0, proc_layout)
    proc_group.setLayout(proc_layout)
    side_panel.addWidget(proc_group)

    self.load_btn = PySide6.QtWidgets.QPushButton("読み込み")
    self.load_btn.clicked.connect(self.load_file)
    side_panel.addWidget(self.load_btn)
    self.save_btn = PySide6.QtWidgets.QPushButton("保存")
    self.save_btn.clicked.connect(self.save_file)
    side_panel.addWidget(self.save_btn)
    side_panel.addStretch()
    main_layout.addLayout(side_panel, 1)

    # --- ズーム対応のための構造変更 ---
    self.scroll_area = PySide6.QtWidgets.QScrollArea()
    self.canvas = PySide6.QtWidgets.QLabel()
    self.canvas.setAlignment(PySide6.QtCore.Qt.AlignCenter)
    self.canvas.setStyleSheet("background-color: #333;")
    self.scroll_area.setWidget(self.canvas)
    self.scroll_area.setWidgetResizable(True)
    self.scroll_area.setAlignment(PySide6.QtCore.Qt.AlignCenter)
    main_layout.addWidget(self.scroll_area, 4)

    PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(
        "Ctrl+Z"), self).activated.connect(self.undo)
    PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(
        "Ctrl+Y"), self).activated.connect(self.redo)

  def _create_slider(self, label: str, min_v: int, max_v: int, init_v: int, layout: PySide6.QtWidgets.QVBoxLayout) -> PySide6.QtWidgets.QSlider:
    layout.addWidget(PySide6.QtWidgets.QLabel(label))
    slider = PySide6.QtWidgets.QSlider(PySide6.QtCore.Qt.Horizontal)
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

  def save_undo_state(self):
    if self.raw_image is not None:
      self.undo_stack.append(self.raw_image.copy())
      if len(self.undo_stack) > self.max_undo: self.undo_stack.pop(0)
      self.redo_stack.clear()
      self.is_modified = True

  def undo(self):
    if self.undo_stack and self.raw_image is not None:
      self.redo_stack.append(self.raw_image.copy())
      self.raw_image = self.undo_stack.pop()
      self.apply_effects()

  def redo(self):
    if self.redo_stack and self.raw_image is not None:
      self.undo_stack.append(self.raw_image.copy())
      self.raw_image = self.redo_stack.pop()
      self.apply_effects()

  def toggle_eraser(self):
    self.eraser_mode = self.eraser_btn.isChecked()
    self.eraser_btn.setText(f"消しゴム: {'ON' if self.eraser_mode else 'OFF'}")

  def change_brush_color(self):
    color = PySide6.QtWidgets.QColorDialog.getColor()
    if color.isValid(): self.current_brush_color = (
        color.blue(), color.green(), color.red())

  def create_blank_canvas(self, w: int, h: int):
    self.current_bg_color = (255, 255, 255)
    self.raw_image = np.full((h, w, 3), 255, dtype=np.uint8)
    self.undo_stack.clear()
    self.redo_stack.clear()
    self.is_modified = False
    self.zoom_factor = 1.0
    self.apply_effects()

  def clear_canvas_with_undo(self):
    self.save_undo_state()
    self.raw_image[:] = self.current_bg_color
    self.apply_effects()

  def load_file(self) -> None:
    path, _ = PySide6.QtWidgets.QFileDialog.getOpenFileName(
        self, "画像選択", "", "Images (*.png *.jpg *.jpeg)")
    if path:
      self.save_undo_state()
      file_bytes = np.fromfile(path, np.uint8)
      self.raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
      self.is_modified = True
      self.zoom_factor = 1.0
      self.apply_effects()

  def save_file(self) -> bool:
    if self.raw_image is None: return False
    path, _ = PySide6.QtWidgets.QFileDialog.getSaveFileName(
        self, "画像を保存", "output.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
    if path:
      ext = ".png" if path.lower().endswith(".png") else ".jpg"
      _, res = cv2.imencode(ext, self.raw_image)
      res.tofile(path)
      self.is_modified = False
      return True
    return False

  def wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None:
    if event.modifiers() & PySide6.QtCore.Qt.ControlModifier:
      # マウスの現在地（キャンバス上の相対座標）を記録
      mouse_pos = event.position()
      old_zoom = self.zoom_factor

      # 倍率計算
      delta = event.angleDelta().y()
      zoom_step = 1.1 if delta > 0 else 1 / 1.1
      self.zoom_factor = max(0.1, min(self.zoom_factor * zoom_step, 10.0))

      # 実際の倍率比を計算
      actual_ratio = self.zoom_factor / old_zoom

      # 再描画
      self.apply_effects()

      # スクロール位置の調整（マウス位置を固定）
      h_bar = self.scroll_area.horizontalScrollBar()
      v_bar = self.scroll_area.verticalScrollBar()

      new_h = (mouse_pos.x() + h_bar.value()) * \
          actual_ratio - mouse_pos.x()
      new_v = (mouse_pos.y() + v_bar.value()) * \
          actual_ratio - mouse_pos.y()

      h_bar.setValue(int(new_h))
      v_bar.setValue(int(new_v))
    else:
      super().wheelEvent(event)

  def mousePressEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
    if event.button() == PySide6.QtCore.Qt.LeftButton:
      pos = self.get_canvas_coordinates(event.pos())
      if not pos: return
      self.save_undo_state()
      if self.draw_mode == 3 and self.raw_image is not None:
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color
        h, w = self.raw_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(self.raw_image, mask, (pos.x(), pos.y()), color)
        self.apply_effects()
        return
      self.last_point = pos
      self.start_point = pos
      self.last_time = time.time()
      self.current_velocity_size = self.brush_size_slider.value()

  def mouseMoveEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
    if not (event.buttons() & PySide6.QtCore.Qt.LeftButton) or not self.start_point or self.raw_image is None: return
    if self.draw_mode == 3: return
    current_point = self.get_canvas_coordinates(event.pos())
    if not current_point: return
    color = self.current_bg_color if self.eraser_mode else self.current_brush_color
    thickness = self.brush_size_slider.value()
    if self.draw_mode == 0:
      if self.last_point:
        if self.pressure_brush_mode:
          now = time.time()
          dt = now - self.last_time if now - self.last_time > 0 else 0.001
          dist = np.sqrt((current_point.x() - self.last_point.x())
                         ** 2 + (current_point.y() - self.last_point.y())**2)
          velocity = dist / dt
          target_size = thickness * \
              max(0.2, min(1.5, 100 / (velocity + 1)))
          self.current_velocity_size = self.current_velocity_size * 0.7 + target_size * 0.3
          draw_thickness = int(self.current_velocity_size)
          self.last_time = now
        else: draw_thickness = thickness
        cv2.line(self.raw_image, (self.last_point.x(), self.last_point.y(
        )), (current_point.x(), current_point.y()), color, max(1, draw_thickness))
        self.last_point = current_point
        self.apply_effects()
    else: self.apply_effects(preview_pos=current_point)

  def mouseReleaseEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
    if event.button() == PySide6.QtCore.Qt.LeftButton and self.start_point and self.raw_image is not None:
      if self.draw_mode == 3:
        self.last_point = self.start_point = None
        return
      end_point = self.get_canvas_coordinates(event.pos())
      if end_point:
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color
        thickness = self.brush_size_slider.value()
        if self.draw_mode == 1: cv2.rectangle(self.raw_image, (self.start_point.x(
        ), self.start_point.y()), (end_point.x(), end_point.y()), color, thickness)
        elif self.draw_mode == 2:
          center = (self.start_point.x(), self.start_point.y())
          radius = int(
              np.sqrt((end_point.x() - center[0])**2 + (end_point.y() - center[1])**2))
          cv2.circle(self.raw_image, center, radius, color, thickness)
        self.apply_effects()
    self.last_point = self.start_point = None

  def get_canvas_coordinates(self, pos: PySide6.QtCore.QPoint) -> typing.Optional[PySide6.QtCore.QPoint]:
    if self.canvas.pixmap() is None or self.raw_image is None: return None
    # QScrollArea内の座標に変換
    local_pos = self.canvas.mapFromParent(
        self.scroll_area.viewport().mapFromParent(pos))

    img_h, img_w = self.raw_image.shape[:2]
    pix_w, pix_h = self.canvas.pixmap().width(), self.canvas.pixmap().height()

    offset_x = (self.canvas.width() - pix_w) / 2
    offset_y = (self.canvas.height() - pix_h) / 2

    px = (local_pos.x() - offset_x) * (img_w / pix_w)
    py = (local_pos.y() - offset_y) * (img_h / pix_h)

    angle = self.rotate_slider.value()
    if angle != 0:
      matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
      inv_matrix = cv2.invertAffineTransform(matrix)
      original_point = inv_matrix @ np.array([px, py, 1.0])
      px, py = original_point[0], original_point[1]

    x, y = int(px), int(py)
    if 0 <= x < img_w and 0 <= y < img_h: return PySide6.QtCore.QPoint(x, y)
    return None

  def apply_effects(self, preview_pos: typing.Optional[PySide6.QtCore.QPoint] = None) -> None:
    if self.raw_image is None: return
    img = self.raw_image.copy()
    color = self.current_bg_color if self.eraser_mode else self.current_brush_color
    thickness = self.brush_size_slider.value()

    if preview_pos and self.start_point:
      if self.draw_mode == 1: cv2.rectangle(img, (self.start_point.x(
      ), self.start_point.y()), (preview_pos.x(), preview_pos.y()), color, thickness)
      elif self.draw_mode == 2:
        center = (self.start_point.x(), self.start_point.y())
        radius = int(
            np.sqrt((preview_pos.x() - center[0])**2 + (preview_pos.y() - center[1])**2))
        cv2.circle(img, center, radius, color, thickness)

    effect_type = self.effect_slider.value()
    if effect_type == 1: img = cv2.cvtColor(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif effect_type == 2: img = cv2.transform(img, np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
    elif effect_type == 3: img = cv2.cvtColor(cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8), cv2.COLOR_GRAY2BGR)

    k = self.blur_slider.value()
    if k > 1: img = cv2.GaussianBlur(
        img, (k if k % 2 != 0 else k + 1, k if k % 2 != 0 else k + 1), 0)

    angle = self.rotate_slider.value()
    if angle != 0:
      h, w = img.shape[:2]
      img = cv2.warpAffine(img, cv2.getRotationMatrix2D(
          (w / 2, h / 2), angle, 1.0), (w, h), borderValue=self.current_bg_color)

    self.display_image(img)

  def display_image(self, img: np.ndarray) -> None:
    h, w, ch = img.shape
    q_img = PySide6.QtGui.QImage(
        img.data, w, h, ch * w, PySide6.QtGui.QImage.Format_BGR888)
    pixmap = PySide6.QtGui.QPixmap.fromImage(q_img)

    # ズーム倍率をピクセルサイズに反映
    scaled_pixmap = pixmap.scaled(pixmap.size() * self.zoom_factor,
                                  PySide6.QtCore.Qt.KeepAspectRatio,
                                  PySide6.QtCore.Qt.SmoothTransformation)
    self.canvas.setPixmap(scaled_pixmap)
    # ラベル自体のサイズを固定してスクロールを発生させる
    self.canvas.setFixedSize(scaled_pixmap.size())

  def closeEvent(self, event: PySide6.QtGui.QCloseEvent) -> None:
    if not self.is_modified: event.accept()
    else:
      reply = PySide6.QtWidgets.QMessageBox.question(
          self, "確認", "終了する前に保存しますか？", PySide6.QtWidgets.QMessageBox.Save | PySide6.QtWidgets.QMessageBox.Discard | PySide6.QtWidgets.QMessageBox.Cancel)
      if reply == PySide6.QtWidgets.QMessageBox.Save:
        if self.save_file(): event.accept()
        else: event.ignore()
      elif reply == PySide6.QtWidgets.QMessageBox.Discard: event.accept()
      else: event.ignore()

if __name__ == "__main__":
  app = PySide6.QtWidgets.QApplication(sys.argv)
  window = AdvancedImageApp()
  window.show()
  sys.exit(app.exec())
