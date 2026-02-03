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
    # 編集中のメイン画像データ(OpenCV/numpy形式)
    self.raw_image: typing.Optional[np.ndarray] = None
    # 最後にマウスがあった座標
    self.last_point: typing.Optional[PySide6.QtCore.QPoint] = None
    # マウスを押した瞬間の座標（直線用）
    self.start_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.eraser_mode = False          # 消しゴムモードのフラグ
    self.pressure_brush_mode = False  # 筆圧（速度連動）ブラシのフラグ
    self.current_bg_color = (255, 255, 255)  # キャンバスの背景色 (BGR形式)
    self.current_brush_color = (0, 0, 0)     # ブラシの色 (BGR形式)

    # --- 速度（筆圧）計算用の変数 ---
    self.last_time = 0.0              # 直前の描画時刻
    self.current_velocity_size = 5.0  # 計算後の現在のブラシ太さ

    # --- Undo/Redo用スタック ---
    self.undo_stack: typing.List[np.ndarray] = []  # 過去の画像状態を保持
    self.redo_stack: typing.List[np.ndarray] = []  # 取り消した状態を保持
    self.max_undo = 30  # 最大履歴数

    self.init_ui()
    self.create_blank_canvas(800, 600)  # 初期キャンバスの作成

  def init_ui(self) -> None:
    """UIコンポーネントの配置と設定"""
    central_widget = PySide6.QtWidgets.QWidget()
    self.setCentralWidget(central_widget)
    main_layout = PySide6.QtWidgets.QHBoxLayout(central_widget)

    # 左側の操作パネル
    side_panel = PySide6.QtWidgets.QVBoxLayout()

    # --- ブラシ・消しゴム設定グループ ---
    draw_group = PySide6.QtWidgets.QGroupBox("ブラシ・消しゴム設定")
    draw_layout = PySide6.QtWidgets.QVBoxLayout()

    self.brush_size_slider = self._create_slider(
        "基準サイズ", 1, 50, 5, draw_layout)

    self.brush_mode_btn = PySide6.QtWidgets.QPushButton("通常ブラシ")
    self.brush_mode_btn.setCheckable(True)  # ON/OFF切り替え可能にする
    self.brush_mode_btn.clicked.connect(self.toggle_brush_mode)
    draw_layout.addWidget(self.brush_mode_btn)

    self.brush_color_btn = PySide6.QtWidgets.QPushButton("ブラシの色を変える")
    self.brush_color_btn.clicked.connect(self.change_brush_color)
    draw_layout.addWidget(self.brush_color_btn)

    self.eraser_btn = PySide6.QtWidgets.QPushButton("消しゴム: OFF")
    self.eraser_btn.setCheckable(True)
    self.eraser_btn.clicked.connect(self.toggle_eraser)
    draw_layout.addWidget(self.eraser_btn)

    # Undo/Redoボタン
    undo_redo_layout = PySide6.QtWidgets.QHBoxLayout()
    self.undo_btn = PySide6.QtWidgets.QPushButton("Undo")
    self.undo_btn.clicked.connect(self.undo)
    self.redo_btn = PySide6.QtWidgets.QPushButton("Redo")
    self.redo_btn.clicked.connect(self.redo)
    undo_redo_layout.addWidget(self.undo_btn)
    undo_redo_layout.addWidget(self.redo_btn)
    draw_layout.addLayout(undo_redo_layout)

    self.clear_btn = PySide6.QtWidgets.QPushButton("キャンバスをクリア")
    self.clear_btn.clicked.connect(self.clear_canvas_with_undo)
    draw_layout.addWidget(self.clear_btn)

    self.bg_color_btn = PySide6.QtWidgets.QPushButton("背景色を変える")
    self.bg_color_btn.clicked.connect(self.change_bg_color)
    draw_layout.addWidget(self.bg_color_btn)

    draw_group.setLayout(draw_layout)
    side_panel.addWidget(draw_group)

    # --- 画像処理エフェクトグループ ---
    proc_group = PySide6.QtWidgets.QGroupBox("画像処理エフェクト")
    proc_layout = PySide6.QtWidgets.QVBoxLayout()

    self.rotate_slider = self._create_slider(
        "回転", -180, 180, 0, proc_layout)
    self.blur_slider = self._create_slider("ぼかし", 1, 51, 1, proc_layout)
    self.effect_slider = self._create_slider(
        "特殊効果 (なし/白黒/セピア/線画)", 0, 3, 0, proc_layout)

    proc_group.setLayout(proc_layout)
    side_panel.addWidget(proc_group)

    # ファイル操作ボタン
    self.load_btn = PySide6.QtWidgets.QPushButton("画像読み込み")
    self.load_btn.clicked.connect(self.load_file)
    side_panel.addWidget(self.load_btn)

    self.save_btn = PySide6.QtWidgets.QPushButton("画像を保存")
    self.save_btn.clicked.connect(self.save_file)
    side_panel.addWidget(self.save_btn)

    side_panel.addStretch()  # 下部に空白を入れてボタンを上に詰める
    main_layout.addLayout(side_panel, 1)

    # --- キャンバス表示エリア ---
    self.canvas = PySide6.QtWidgets.QLabel()
    self.canvas.setAlignment(PySide6.QtCore.Qt.AlignCenter)
    self.canvas.setStyleSheet(
        "background-color: #333; border: 2px solid #555;")
    main_layout.addWidget(self.canvas, 4)

    # ショートカットキー登録
    PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(
        "Ctrl+Z"), self).activated.connect(self.undo)
    PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(
        "Ctrl+Y"), self).activated.connect(self.redo)

  def _create_slider(self, label: str, min_v: int, max_v: int, init_v: int, layout: PySide6.QtWidgets.QVBoxLayout) -> PySide6.QtWidgets.QSlider:
    """スライダーとラベルをセットで作成しレイアウトに追加する"""
    layout.addWidget(PySide6.QtWidgets.QLabel(label))
    slider = PySide6.QtWidgets.QSlider(PySide6.QtCore.Qt.Horizontal)
    slider.setRange(min_v, max_v)
    slider.setValue(init_v)
    slider.valueChanged.connect(self.apply_effects)  # 値が変わったらリアルタイムでエフェクト反映
    layout.addWidget(slider)
    return slider

  def toggle_brush_mode(self):
    """通常ブラシと筆圧ブラシを切り替える"""
    self.pressure_brush_mode = self.brush_mode_btn.isChecked()
    self.brush_mode_btn.setText(
        "筆圧ブラシ" if self.pressure_brush_mode else "通常ブラシ")

  def save_undo_state(self):
    """現在の画像状態をスタックに保存（描画開始時やエフェクト変更前に呼ぶ）"""
    if self.raw_image is not None:
      self.undo_stack.append(self.raw_image.copy())
      if len(self.undo_stack) > self.max_undo:
        self.undo_stack.pop(0)  # 履歴制限を超えたら古いものを消す
      self.redo_stack.clear()  # 新しい操作をしたらRedo履歴は消去

  def undo(self):
    """一つ前の状態に戻す"""
    if self.undo_stack and self.raw_image is not None:
      self.redo_stack.append(self.raw_image.copy())
      self.raw_image = self.undo_stack.pop()
      self.apply_effects()

  def redo(self):
    """戻した操作をやり直す"""
    if self.redo_stack and self.raw_image is not None:
      self.undo_stack.append(self.raw_image.copy())
      self.raw_image = self.redo_stack.pop()
      self.apply_effects()

  def toggle_eraser(self):
    """消しゴムモードのON/OFF"""
    self.eraser_mode = self.eraser_btn.isChecked()
    self.eraser_btn.setText(f"消しゴム: {'ON' if self.eraser_mode else 'OFF'}")

  def change_brush_color(self):
    """カラーダイアログでブラシの色を選択"""
    color = PySide6.QtWidgets.QColorDialog.getColor()
    if color.isValid():
      self.current_brush_color = (
          color.blue(), color.green(), color.red())  # RGB -> BGR

  def change_bg_color(self):
    """カラーダイアログで背景色を選択し、キャンバス全体を塗りつぶす"""
    color = PySide6.QtWidgets.QColorDialog.getColor()
    if color.isValid() and self.raw_image is not None:
      self.save_undo_state()
      self.current_bg_color = (color.blue(), color.green(), color.red())
      self.raw_image[:] = self.current_bg_color
      self.apply_effects()

  def create_blank_canvas(self, w: int, h: int):
    """真っ白なキャンバスを新規作成"""
    self.current_bg_color = (255, 255, 255)
    self.raw_image = np.full((h, w, 3), 255, dtype=np.uint8)
    self.undo_stack.clear()
    self.redo_stack.clear()
    self.apply_effects()

  def clear_canvas_with_undo(self):
    """キャンバスを現在の背景色でリセット（Undo可能）"""
    self.save_undo_state()
    self.raw_image[:] = self.current_bg_color
    self.apply_effects()

  def load_file(self) -> None:
    """画像をファイルから読み込む（日本語パス対応のためnp.fromfileを使用）"""
    path, _ = PySide6.QtWidgets.QFileDialog.getOpenFileName(
        self, "画像選択", "", "Images (*.png *.jpg *.jpeg)")
    if path:
      self.save_undo_state()
      file_bytes = np.fromfile(path, np.uint8)
      self.raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
      # 読み込んだ画像の左上の色を暫定的な背景色とする
      self.current_bg_color = tuple(map(int, self.raw_image[0, 0]))
      self.apply_effects()

  def save_file(self) -> None:
    """現在の画像を保存（エフェクト適用前の生データを保存）"""
    if self.raw_image is None: return
    path, _ = PySide6.QtWidgets.QFileDialog.getSaveFileName(
        self, "画像を保存", "output.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
    if path:
      ext = ".png" if path.lower().endswith(".png") else ".jpg"
      _, res = cv2.imencode(ext, self.raw_image)
      res.tofile(path)

  # --- マウスイベント ---
  def mousePressEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
    if event.button() == PySide6.QtCore.Qt.LeftButton:
      self.save_undo_state()
      pos = self.get_canvas_coordinates(event.pos())
      self.last_point = pos
      self.start_point = pos
      self.last_time = time.time()
      self.current_velocity_size = self.brush_size_slider.value()

  def mouseMoveEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
    """マウス移動中の描画処理"""
    if event.buttons() & PySide6.QtCore.Qt.LeftButton and self.last_point and not (event.modifiers() & PySide6.QtCore.Qt.ShiftModifier):
      current_point = self.get_canvas_coordinates(event.pos())
      if current_point and self.raw_image is not None:
        base_size = self.brush_size_slider.value()
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color

        if self.pressure_brush_mode:
          # 速度に基づく太さの動的変化計算
          now = time.time()
          dt = now - self.last_time if now - self.last_time > 0 else 0.001
          dist = np.sqrt((current_point.x() - self.last_point.x())
                         ** 2 + (current_point.y() - self.last_point.y())**2)
          velocity = dist / dt
          # 速いほど細く、遅いほど太くする。100/(v+1)で係数を算出
          target_size = base_size * \
              max(0.2, min(1.5, 100 / (velocity + 1)))
          # 線が急に太くなったり細くなったりしないようスムージング
          self.current_velocity_size = self.current_velocity_size * 0.7 + target_size * 0.3
          draw_size = int(self.current_velocity_size)
          self.last_time = now
        else:
          draw_size = base_size

        # 直前の点と現在の点を結ぶ（OpenCVのlineは高速）
        cv2.line(self.raw_image, (self.last_point.x(), self.last_point.y()),
                 (current_point.x(), current_point.y()), color, max(1, draw_size))
        self.last_point = current_point
        self.apply_effects()  # 画面更新

  def mouseReleaseEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
    """マウスを離した時の処理（Shift+クリックでの直線描画）"""
    if event.button() == PySide6.QtCore.Qt.LeftButton and (event.modifiers() & PySide6.QtCore.Qt.ShiftModifier):
      end_point = self.get_canvas_coordinates(event.pos())
      if self.start_point and end_point and self.raw_image is not None:
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color
        cv2.line(self.raw_image, (self.start_point.x(), self.start_point.y()),
                 (end_point.x(), end_point.y()), color, self.brush_size_slider.value())
        self.apply_effects()
    self.last_point = None
    self.start_point = None

  def get_canvas_coordinates(self, pos: PySide6.QtCore.QPoint) -> typing.Optional[PySide6.QtCore.QPoint]:
    """ウィジェット上のマウス座標を、実際の画像上のピクセル座標に変換（回転・比率対応）"""
    if self.canvas.pixmap() is None or self.raw_image is None: return None
    lbl_w, lbl_h = self.canvas.width(), self.canvas.height()
    img_h, img_w = self.raw_image.shape[:2]

    # 表示比率とオフセットの計算
    ratio = min(lbl_w / img_w, lbl_h / img_h)
    offset_x = (lbl_w - img_w * ratio) / 2
    offset_y = (lbl_h - img_h * ratio) / 2

    # ラベル内の相対座標へ変換
    rel_x = pos.x() - self.canvas.x() - offset_x
    rel_y = pos.y() - self.canvas.y() - offset_y

    # 画像スケールに変換
    px, py = rel_x / ratio, rel_y / ratio

    # 回転がかかっている場合、座標を逆回転させて「元の画像での位置」を特定する
    angle = self.rotate_slider.value()
    if angle != 0:
      matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
      inv_matrix = cv2.invertAffineTransform(matrix)
      point = np.array([px, py, 1.0])
      original_point = inv_matrix @ point
      px, py = original_point[0], original_point[1]

    x, y = int(px), int(py)
    # 画像の範囲外ならNoneを返す
    if 0 <= x < img_w and 0 <= y < img_h: return PySide6.QtCore.QPoint(x, y)
    return None

  def apply_effects(self) -> None:
    """生データ(raw_image)にスライダーの設定値を反映させて表示用画像を生成"""
    if not hasattr(self, 'effect_slider') or self.raw_image is None: return
    img = self.raw_image.copy()

    # 1. 特殊エフェクト
    effect_type = self.effect_slider.value()
    if effect_type == 1:  # グレースケール
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif effect_type == 2:  # セピア
      kernel = np.array([[0.272, 0.534, 0.131],
                         [0.349, 0.686, 0.168],
                         [0.393, 0.769, 0.189]])
      img = cv2.transform(img, kernel)  # 行列演算で色変換
    elif effect_type == 3:  # エッジ抽出 (線画化)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      edge = cv2.adaptiveThreshold(
          gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
      img = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # 2. ぼかし (奇数のカーネルサイズが必要)
    k = self.blur_slider.value()
    if k > 1:
      k = k if k % 2 != 0 else k + 1
      img = cv2.GaussianBlur(img, (k, k), 0)

    # 3. 回転 (アフィン変換)
    angle = self.rotate_slider.value()
    if angle != 0:
      h, w = img.shape[:2]
      matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
      img = cv2.warpAffine(img, matrix, (w, h),
                           borderValue=self.current_bg_color)

    self.display_image(img)

  def display_image(self, img: np.ndarray) -> None:
    """OpenCV形式の画像をQPixmapに変換してQLabelに表示"""
    h, w, ch = img.shape
    bytes_per_line = ch * w
    q_img = PySide6.QtGui.QImage(
        img.data, w, h, bytes_per_line, PySide6.QtGui.QImage.Format_BGR888)
    pixmap = PySide6.QtGui.QPixmap.fromImage(q_img)

    # キャンバスのサイズに合わせてリサイズして表示
    self.canvas.setPixmap(pixmap.scaled(
        self.canvas.size(),
        PySide6.QtCore.Qt.KeepAspectRatio,
        PySide6.QtCore.Qt.SmoothTransformation
    ))

if __name__ == "__main__":
  app = PySide6.QtWidgets.QApplication(sys.argv)
  window = AdvancedImageApp()
  window.show()
  sys.exit(app.exec())
