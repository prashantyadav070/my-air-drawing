import cv2
import mediapipe as mp
import numpy as np

# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

canvas, prev_cursor, prev_draw = None, None, None
draw_color, brush_size = (255, 0, 255), 8
eraser_btn = (25, 25, 180, 85)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape
  if canvas is None:
    canvas = np.zeros_like(frame)

  # Draw Eraser Button UI
  x1, y1, x2, y2 = eraser_btn
  cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), cv2.FILLED)
  cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 255, 170), 2)
  cv2.putText(
      frame,
      "ERASE",
      (x1 + 28, y1 + 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.9,
      (90, 255, 170),
      2,
  )

  results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

  if results.multi_hand_landmarks:
    for lm in results.multi_hand_landmarks:
      mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
      lms = lm.landmark

      # Coordinates
      tx, ty = int(lms[4].x * w), int(lms[4].y * h)
      ix, iy = int(lms[8].x * w), int(lms[8].y * h)
      wx, wy = int(lms[0].x * w), int(lms[0].y * h)
      mx, my = int(lms[9].x * w), int(lms[9].y * h)

      # Smoothing
      if prev_cursor is None:
        sx, sy = ix, iy
      else:
        sx = int(prev_cursor[0] + (ix - prev_cursor[0]) * 0.45)
        sy = int(prev_cursor[1] + (iy - prev_cursor[1]) * 0.45)
      prev_cursor = (sx, sy)

      # Modes & Actions
      draw_mode = np.hypot(ix - tx, iy - ty) < np.hypot(mx - wx, my - wy) * 0.55

      if x1 <= sx <= x2 and y1 <= sy <= y2:
        canvas, prev_draw, draw_mode = np.zeros_like(frame), None, False

      # Cursor
      cv2.circle(
          frame,
          (sx, sy),
          brush_size + 4,
          (90, 255, 170) if draw_mode else (180, 180, 180),
          cv2.FILLED,
      )

      # Drawing logic
      if draw_mode:
        if prev_draw is None:
          prev_draw = (sx, sy)
        else:
          if np.hypot(sx - prev_draw[0], sy - prev_draw[1]) < 120:
            cv2.line(canvas, prev_draw, (sx, sy), draw_color, brush_size)
          prev_draw = (sx, sy)
      else:
        prev_draw = None
  else:
    prev_cursor, prev_draw = None, None

  # Output & Controls
  cv2.imshow("Air Draw", cv2.add(frame, canvas))
  key = cv2.waitKey(1) & 0xFF
  if key == ord("c"):
    canvas = np.zeros_like(frame)
  if key == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
