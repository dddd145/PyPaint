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
    self.setMinimumSize(1000, 700)

    # --- 状態保持用の変数 ---
    # 元の画像データ（OpenCV形式: BGR）
    self.raw_image: typing.Optional[np.ndarray] = None
    # 直前のマウス座標
    self.last_point: typing.Optional[PySide6.QtCore.QPoint] = None
    # 描画開始時の座標
    self.start_point: typing.Optional[PySide6.QtCore.QPoint] = None
    self.eraser_mode = False  # 消しゴムモードのフラグ
    self.pressure_brush_mode = False  # 筆圧（速度感応）モードのフラグ
    self.current_bg_color = (255, 255, 255)  # 背景色（白）
    self.current_brush_color = (0, 0, 0)  # ブラシ色（黒）
    self.zoom_factor = 1.0  # 拡大率

    self.draw_mode = 0  # 0:ブラシ, 1:矩形, 2:円, 3:塗りつぶし, 4:選択
    # 選択範囲の座標保持用
    self.selection_rect: typing.Optional[PySide6.QtCore.QRect] = None
    self.clipboard: typing.Optional[np.ndarray] = None  # コピペ用
    self.is_modified = False  # 保存後に変更があったかどうかのフラグ
    self.last_time = 0.0  # 速度計算用の時間記録
    self.current_velocity_size = 5.0  # 現在の筆圧計算によるブラシサイズ

    # 手のひら・スポイト用の変数
    self.is_panning = False  # スペースキー押下中のパン（移動）状態
    self.last_mouse_pos = PySide6.QtCore.QPoint()  # パン移動時の直前座標

    # Undo/Redo管理
    self.undo_stack: typing.List[np.ndarray] = []  # 過去の画像履歴
    self.redo_stack: typing.List[np.ndarray] = []  # やり直した履歴
    self.max_undo = 30  # 最大保存数

    self.init_ui()  # ユーザーインターフェースの初期化
    self.create_blank_canvas(800, 600)  # 空のキャンバス作成

  # UIレイアウトの構築
  def init_ui(self) -> None:
    central_widget = PySide6.QtWidgets.QWidget()
    self.setCentralWidget(central_widget)
    main_layout = PySide6.QtWidgets.QHBoxLayout(central_widget)

    # --- 左側サイドパネル ---
    side_panel = PySide6.QtWidgets.QVBoxLayout()
    draw_group = PySide6.QtWidgets.QGroupBox("ツール設定")
    draw_layout = PySide6.QtWidgets.QVBoxLayout()

    # ブラシサイズ調整スライダー
    self.brush_size_slider = self._create_slider(
        "サイズ", 1, 50, 5, draw_layout)

    # 描画モード選択
    self.mode_combo = PySide6.QtWidgets.QComboBox()
    self.mode_combo.addItems(
        ["通常ブラシ", "矩形 (Rect)", "円 (Circle)", "塗りつぶし (Fill)", "選択 (Select)"])
    self.mode_combo.currentIndexChanged.connect(self.change_draw_mode)
    draw_layout.addWidget(PySide6.QtWidgets.QLabel("描画モード:"))
    draw_layout.addWidget(self.mode_combo)

    # 筆圧感度切り替えボタン
    self.brush_mode_btn = PySide6.QtWidgets.QPushButton("筆圧感度: OFF")
    self.brush_mode_btn.setCheckable(True)
    self.brush_mode_btn.clicked.connect(self.toggle_brush_mode)
    draw_layout.addWidget(self.brush_mode_btn)

    # カラー選択ボタン
    self.brush_color_btn = PySide6.QtWidgets.QPushButton("色を変える")
    self.brush_color_btn.clicked.connect(self.change_brush_color)
    draw_layout.addWidget(self.brush_color_btn)

    # 消しゴム切り替えボタン
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

    # キャンバスクリア
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
    self.effect_slider = self._create_slider("特殊効果", 0, 3, 0, proc_layout)
    proc_group.setLayout(proc_layout)
    side_panel.addWidget(proc_group)

    # --- ファイル操作ボタン ---
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
    self.canvas.setStyleSheet("background-color: #222;")  # 背景は暗灰色
    self.scroll_area.setWidget(self.canvas)
    self.scroll_area.setWidgetResizable(True)
    self.scroll_area.setAlignment(
        PySide6.QtCore.Qt.AlignmentFlag.AlignCenter)
    main_layout.addWidget(self.scroll_area, 4)

    # --- ショートカットキーの登録 ---
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
    # --- 編集ショートカット追加 ---
    PySide6.QtGui.QShortcut(
        "Delete", self).activated.connect(self.delete_selection)
    PySide6.QtGui.QShortcut(
        "Ctrl+C", self).activated.connect(self.copy_selection)
    PySide6.QtGui.QShortcut(
        "Ctrl+V", self).activated.connect(self.paste_selection)

  # スライダー作成のヘルパー関数
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

  # 筆圧感度モードの切り替え
  def toggle_brush_mode(self):
    self.pressure_brush_mode = self.brush_mode_btn.isChecked()
    self.brush_mode_btn.setText(
        "筆圧感度: ON" if self.pressure_brush_mode else "筆圧感度: OFF")
  # 消しゴムモードの切り替え

  def toggle_eraser(self):
    self.eraser_mode = self.eraser_btn.isChecked()
    self.eraser_btn.setText(f"消しゴム: {'ON' if self.eraser_mode else 'OFF'}")

  # 新規キャンバス作成
  def create_blank_canvas(self, w, h):
    self.raw_image = np.full((h, w, 3), 255, dtype=np.uint8)  # 白塗りの画像
    self.undo_stack.clear()
    self.redo_stack.clear()
    self.zoom_factor = 1.0
    self.is_modified = False
    self.apply_effects()

  # スクリーン座標からキャンバス上の正確な座標を逆計算する重要関数
  def get_canvas_coordinates(self, pos: PySide6.QtCore.QPoint) -> typing.Optional[PySide6.QtCore.QPoint]:
    if self.raw_image is None or self.canvas.pixmap() is None: return None
    # ウィジェット上のローカル座標を取得
    local_pos = self.canvas.mapFromGlobal(self.mapToGlobal(pos))
    pix_size = self.canvas.pixmap().size()

    # キャンバス内の画像表示オフセット（余白）を計算
    offset_x = (self.canvas.width() - pix_size.width()) / 2
    offset_y = (self.canvas.height() - pix_size.height()) / 2

    # ズーム後の座標系における位置
    view_x = local_pos.x() - offset_x
    view_y = local_pos.y() - offset_y

    # オリジナルの画像解像度に合わせてスケール変換
    img_h, img_w = self.raw_image.shape[:2]
    px = view_x * (img_w / pix_size.width())
    py = view_y * (img_h / pix_size.height())

    # 画像が回転している場合、アフィン変換の逆行列を用いて回転前の座標を特定
    angle = self.rotate_slider.value()
    if angle != 0:
      matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
      inv_matrix = cv2.invertAffineTransform(matrix)
      original_point = inv_matrix @ np.array([px, py, 1.0])
      px, py = original_point[0], original_point[1]

    x, y = int(round(px)), int(round(py))
    # 画像の範囲内に収まっているかチェック
    if 0 <= x < img_w and 0 <= y < img_h: return PySide6.QtCore.QPoint(x, y)
    return None

  # マウスホイールによるズーム処理 (Ctrlキー併用)
  def wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None:
    if event.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.ControlModifier:
      pos_before = event.position()
      # ズームの中心点を特定
      target_pixel = self.get_canvas_coordinates(pos_before.toPoint())
      if event.angleDelta().y() > 0: self.zoom_factor *= 1.1
      else: self.zoom_factor /= 1.1
      self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
      self.apply_effects()
      # ズーム後に元のピクセル位置へスクロールを合わせる
      if target_pixel: self.adjust_scroll_to_pixel(
          target_pixel, pos_before)
    else: super().wheelEvent(event)

  # ズーム中心点を維持するためのスクロール位置調整
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

  # 非破壊エフェクト適用とプレビュー表示
  def apply_effects(self, preview_pos=None) -> None:
    if self.raw_image is None: return
    img = self.raw_image.copy()

    # 確定前の図形を一時的に重ね書き（プレビュー）
    if preview_pos and self.start_point:
      color = self.current_bg_color if self.eraser_mode else self.current_brush_color
      thickness = self.brush_size_slider.value()

      modifiers = PySide6.QtWidgets.QApplication.keyboardModifiers()
      if self.draw_mode == 0 and modifiers & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier:
        cv2.line(img, (self.start_point.x(), self.start_point.y()),
                 (preview_pos.x(), preview_pos.y()), color, thickness)
      elif self.draw_mode == 1:
        cv2.rectangle(img, (self.start_point.x(), self.start_point.y()),
                      (preview_pos.x(), preview_pos.y()), color, thickness)
      elif self.draw_mode == 2:
        center = (self.start_point.x(), self.start_point.y())
        radius = int(np.linalg.norm(
            [preview_pos.x() - center[0], preview_pos.y() - center[1]]))
        cv2.circle(img, center, radius, color, thickness)

    # --- 画像処理エフェクトの計算 ---
    eff = self.effect_slider.value()
    if eff == 1:  # モノクロ
      img = cv2.cvtColor(cv2.cvtColor(
          img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif eff == 2:  # セピア
      kernel = np.array(
          [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
      img = cv2.transform(img, kernel)
    elif eff == 3:  # エッジ抽出
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      edge = cv2.adaptiveThreshold(
          gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
      img = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # ぼかし処理
    k = self.blur_slider.value()
    if k > 1: img = cv2.GaussianBlur(img, (k | 1, k | 1), 0)

    # 回転処理
    angle = self.rotate_slider.value()
    if angle != 0:
      h, w = img.shape[:2]
      M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
      img = cv2.warpAffine(
          img, M, (w, h), borderValue=self.current_bg_color)

    # --- 選択枠の描画（回転エフェクトの後に行うことで枠が回らないようにする） ---
    # ドラッグ中、または確定済みの選択範囲がある場合
    current_sel = None
    if self.draw_mode == 4 and self.start_point and preview_pos:
      current_sel = PySide6.QtCore.QRect(
          self.start_point, preview_pos).normalized()
    elif self.selection_rect:
      current_sel = self.selection_rect

    if current_sel:
      # 回転後の画像における座標を再計算
      h, w = self.raw_image.shape[:2]
      M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
      pts = np.array([
          [current_sel.left(), current_sel.top()],
          [current_sel.right(), current_sel.top()],
          [current_sel.right(), current_sel.bottom()],
          [current_sel.left(), current_sel.bottom()]
      ], dtype='float32')
      # 各頂点を回転行列で変換
      ones = np.ones(shape=(len(pts), 1))
      pts_ones = np.concatenate([pts, ones], axis=1)
      rotated_pts = M.dot(pts_ones.T).T.astype(int)

      # 白と黒の枠線を描画（見やすくするため2重にする）
      cv2.polylines(img, [rotated_pts], True, (0, 0, 0), 2)
      cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 1, cv2.LINE_4)

    self.display_image(img)

  # OpenCV画像をQt画像に変換して画面表示
  def display_image(self, img):
    h, w, ch = img.shape
    q_img = PySide6.QtGui.QImage(
        img.data, w, h, ch * w, PySide6.QtGui.QImage.Format.Format_BGR888)
    pix = PySide6.QtGui.QPixmap.fromImage(q_img)
    # ズーム倍率を反映して拡大縮小
    scaled_pix = pix.scaled(pix.size() * self.zoom_factor,
                            PySide6.QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            PySide6.QtCore.Qt.TransformationMode.SmoothTransformation)
    self.canvas.setPixmap(scaled_pix)
    self.canvas.setFixedSize(scaled_pix.size())

  # パン機能の開始判定（スペースキー）
  def keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
    if event.key() == PySide6.QtCore.Qt.Key.Key_Space:
      if not event.isAutoRepeat():
        self.is_panning = True
        self.canvas.setCursor(
            PySide6.QtCore.Qt.CursorShape.ClosedHandCursor)
    super().keyPressEvent(event)

  # パン機能の終了判定
  def keyReleaseEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
    if event.key() == PySide6.QtCore.Qt.Key.Key_Space:
      if not event.isAutoRepeat():
        self.is_panning = False
        self.canvas.setCursor(PySide6.QtCore.Qt.CursorShape.ArrowCursor)
    super().keyReleaseEvent(event)

  # マウス押下時のイベント
  def mousePressEvent(self, event):
    # 1. パン移動中の場合
    if self.is_panning:
      self.last_mouse_pos = event.pos()
      return

    # 2. 座標取得
    pos = self.get_canvas_coordinates(event.pos())
    if not pos:
      # キャンバス外クリックで選択解除
      self.selection_rect = None
      self.apply_effects()
      return

    # 3. スポイト機能 (Altキー押下時)
    if event.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.AltModifier:
      bgr = self.raw_image[pos.y(), pos.x()]
      self.current_brush_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
      return

    # 4. 通常の描画処理開始
    # 選択モード以外でクリックされたら選択を解除
    if self.draw_mode != 4:
      self.selection_rect = None

    self.save_undo_state()
    if self.draw_mode == 3:  # 塗りつぶし
      color = self.current_bg_color if self.eraser_mode else self.current_brush_color
      cv2.floodFill(self.raw_image, None, (pos.x(), pos.y()), color)
      self.apply_effects()
    else:  # ブラシ/矩形/円/選択の開始点
      self.start_point = self.last_point = pos
      self.last_time = time.time()

  # マウスドラッグ中のイベント
  def mouseMoveEvent(self, event):
    # パン移動（手のひらツール）の処理
    if self.is_panning:
      delta = event.pos() - self.last_mouse_pos
      self.last_mouse_pos = event.pos()
      h_bar = self.scroll_area.horizontalScrollBar()
      v_bar = self.scroll_area.verticalScrollBar()
      h_bar.setValue(h_bar.value() - delta.x())
      v_bar.setValue(v_bar.value() - delta.y())
      return

    # スポイト中は描画しない
    if event.modifiers() & PySide6.QtCore.Qt.KeyboardModifier.AltModifier:
      return

    if not (event.buttons() & PySide6.QtCore.Qt.MouseButton.LeftButton) or not self.start_point: return
    pos = self.get_canvas_coordinates(event.pos())
    if not pos: return

    if self.draw_mode == 0:  # 通常ブラシ描画
      modifiers = event.modifiers()
      if modifiers & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier:
        # Shift押下時は直線のプレビュー
        self.apply_effects(preview_pos=pos)
      else:
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color
        thick = self.brush_size_slider.value()
        # 筆圧（速度）モードの計算
        if self.pressure_brush_mode:
          now = time.time()
          dist = np.linalg.norm(
              [pos.x() - self.last_point.x(), pos.y() - self.last_point.y()])
          v = dist / (now - self.last_time + 0.001)
          # 速度に合わせてサイズを変動させる（ローパスフィルタで滑らかに）
          self.current_velocity_size = self.current_velocity_size * \
              0.8 + (thick * max(0.2, min(1.2, 80 / (v + 1)))) * 0.2
          thick = int(self.current_velocity_size)
          self.last_time = now
        # 実際の描画を元画像(raw_image)に書き込む
        cv2.line(self.raw_image, (self.last_point.x(), self.last_point.y(
        )), (pos.x(), pos.y()), color, max(1, thick))
        self.last_point = pos
        self.apply_effects()
    else:  # 図形・選択モードはプレビューを更新するだけ
      self.apply_effects(preview_pos=pos)

  # マウス離した時のイベント（図形の確定）
  def mouseReleaseEvent(self, event):
    if self.is_panning: return

    if self.start_point:
      pos = self.get_canvas_coordinates(event.pos())
      if pos:
        color = self.current_bg_color if self.eraser_mode else self.current_brush_color
        thick = self.brush_size_slider.value()
        modifiers = event.modifiers()

        # 図形の種類に応じて元画像に最終的な書き込みを行う
        if self.draw_mode == 0 and modifiers & PySide6.QtCore.Qt.KeyboardModifier.ShiftModifier:
          cv2.line(self.raw_image, (self.start_point.x(
          ), self.start_point.y()), (pos.x(), pos.y()), color, thick)
        elif self.draw_mode == 1:  # 矩形確定
          cv2.rectangle(self.raw_image, (self.start_point.x(
          ), self.start_point.y()), (pos.x(), pos.y()), color, thick)
        elif self.draw_mode == 2:  # 円確定
          center = (self.start_point.x(), self.start_point.y())
          radius = int(np.linalg.norm(
              [pos.x() - center[0], pos.y() - center[1]]))
          cv2.circle(self.raw_image, center, radius, color, thick)
        # --- 選択確定 ---
        elif self.draw_mode == 4:
          self.selection_rect = PySide6.QtCore.QRect(
              self.start_point, pos).normalized()
        self.apply_effects()
    self.start_point = self.last_point = None

  # --- 選択範囲の操作機能 ---
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
      # 貼り付け位置は現在のマウス位置か、なければ左上
      target = self.last_point if self.last_point else PySide6.QtCore.QPoint(
          0, 0)
      x2, y2 = min(self.raw_image.shape[1], target.x(
      ) + w), min(self.raw_image.shape[0], target.y() + h)
      self.raw_image[target.y():y2, target.x(
      ):x2] = self.clipboard[:y2 - target.y(), :x2 - target.x()]
      # 貼り付けた範囲を新しい選択範囲にする
      self.selection_rect = PySide6.QtCore.QRect(
          target.x(), target.y(), x2 - target.x(), y2 - target.y())
      self.apply_effects()

  # Undo用のスタック管理
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
    self.raw_image[:] = 255  # 全体を白で塗りつぶし
    self.apply_effects()

  # 画像ファイルの読み込み
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

  # 画像ファイルの保存
  def save_file(self):
    path, _ = PySide6.QtWidgets.QFileDialog.getSaveFileName(
        self, "画像を保存", "output.png", "PNG (*.png)")
    if path:
      cv2.imencode(".png", self.raw_image)[1].tofile(path)
      self.is_modified = False
      return True
    return False

  # アプリ終了時の未保存確認
  def closeEvent(self, event: PySide6.QtGui.QCloseEvent):
    if not self.is_modified:
      event.accept()
      return
    reply = PySide6.QtWidgets.QMessageBox.question(
        self, "確認", "変更が保存されていません。終了する前に保存しますか？",
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
