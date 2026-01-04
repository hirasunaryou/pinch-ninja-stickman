"""
Pinch Ninja Stickman
--------------------
Single-file webcam mini-game built for beginners. Uses only OpenCV, MediaPipe, and NumPy.

How to play:
- Make a pinch with your thumb and index finger to swing the stickman's sword.
- Hit the bouncing targets before the 30-second timer ends.
- Press 'r' to restart or 'q' to quit at any time.
"""

# ============================= EDIT HERE =============================
# Friendly tuning zone. Adjust these values to change the game feel.
PINCH_THRESHOLD = 0.07         # Smaller => pinch triggers sooner (normalized landmark distance)
HOLD_FRAMES = 3                # Frames required to turn pinch ON (debounce)
RELEASE_FRAMES = 3             # Frames required to turn pinch OFF (debounce)
TARGET_COUNT = 8               # Number of bouncing targets
TARGET_RADIUS = 18             # Target circle radius in pixels
TARGET_SPEED_RANGE = (3, 6)    # Min/max random speed in pixels per frame
SLASH_LENGTH = 140             # Visual slash length from the stickman's right hand
SLASH_SWEEP_DEG = 80           # How wide the slash arc swings (degrees)
HIT_RADIUS = 28                # Distance from slash line that counts as a hit (pixels)
SCORE_PER_HIT = 10             # Points per target hit
GAME_DURATION = 30             # Seconds per round
WINDOW_NAME = "Pinch Ninja Stickman"
# ====================================================================

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


# Helper dataclass to store target state.
@dataclass
class Target:
    position: np.ndarray  # shape (2,) as [x, y]
    velocity: np.ndarray  # shape (2,) as [vx, vy]

    def move(self, width: int, height: int) -> None:
        """Move the target and bounce off window borders."""
        self.position += self.velocity
        # Bounce horizontally
        if self.position[0] <= TARGET_RADIUS or self.position[0] >= width - TARGET_RADIUS:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], TARGET_RADIUS, width - TARGET_RADIUS)
        # Bounce vertically
        if self.position[1] <= TARGET_RADIUS or self.position[1] >= height - TARGET_RADIUS:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], TARGET_RADIUS, height - TARGET_RADIUS)

    def respawn(self, width: int, height: int) -> None:
        """Place the target at a new random position and velocity."""
        self.position = np.array([
            random.randint(TARGET_RADIUS, width - TARGET_RADIUS),
            random.randint(TARGET_RADIUS, height - TARGET_RADIUS),
        ], dtype=np.float32)
        speed = random.uniform(*TARGET_SPEED_RANGE)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * speed


def create_targets(width: int, height: int) -> List[Target]:
    """Spawn a list of targets with random positions and velocities."""
    targets: List[Target] = []
    for _ in range(TARGET_COUNT):
        t = Target(position=np.zeros(2, dtype=np.float32), velocity=np.zeros(2, dtype=np.float32))
        t.respawn(width, height)
        targets.append(t)
    return targets


def draw_stickman(frame: np.ndarray, center: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Draw a simple stickman and return left/right hand positions."""
    cx, cy = center
    head_r = 25
    body_len = 70
    arm_len = 55
    leg_len = 60

    # Head and body
    cv2.circle(frame, (cx, cy - body_len), head_r, (0, 0, 0), 2)
    cv2.line(frame, (cx, cy - body_len + head_r), (cx, cy + body_len), (0, 0, 0), 2)

    # Arms
    left_hand = (cx - arm_len, cy - 10)
    right_hand = (cx + arm_len, cy - 10)
    cv2.line(frame, (cx - 10, cy - 10), left_hand, (0, 0, 0), 2)
    cv2.line(frame, (cx + 10, cy - 10), right_hand, (0, 0, 0), 2)

    # Legs
    cv2.line(frame, (cx, cy + body_len), (cx - 30, cy + body_len + leg_len), (0, 0, 0), 2)
    cv2.line(frame, (cx, cy + body_len), (cx + 30, cy + body_len + leg_len), (0, 0, 0), 2)

    return left_hand, right_hand


def slash_line(right_hand: Tuple[int, int], frame_count: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Create an animated slash line that sweeps over time."""
    # Sweep angle oscillates to feel alive.
    base_angle_deg = 20
    sweep = SLASH_SWEEP_DEG * math.sin(frame_count / 7.0)
    angle_rad = math.radians(base_angle_deg + sweep)
    start = np.array(right_hand, dtype=np.float32)
    direction = np.array([math.cos(angle_rad), -math.sin(angle_rad)], dtype=np.float32)
    end = start + direction * SLASH_LENGTH
    return tuple(start.astype(int)), tuple(end.astype(int))


def point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Shortest distance from a point to a line segment."""
    line = end - start
    line_len_sq = float(np.dot(line, line))
    if line_len_sq == 0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, line) / line_len_sq)
    t = max(0.0, min(1.0, t))
    projection = start + t * line
    return float(np.linalg.norm(point - projection))


def check_hits(targets: List[Target], slash_start: Tuple[int, int], slash_end: Tuple[int, int]) -> List[int]:
    """Return indices of targets hit by the slash."""
    hits = []
    start = np.array(slash_start, dtype=np.float32)
    end = np.array(slash_end, dtype=np.float32)
    for idx, t in enumerate(targets):
        dist = point_line_distance(t.position, start, end)
        if dist <= HIT_RADIUS:
            hits.append(idx)
    return hits


def draw_hud(frame: np.ndarray, score: int, remaining: float, pinch_on: bool, pinch_dist: float, fps: float) -> None:
    """Draw heads-up display with score, timer, pinch info, and FPS."""
    hud_color = (20, 40, 180)
    pinch_text = "ON" if pinch_on else "OFF"
    cv2.rectangle(frame, (10, 10), (330, 140), (255, 255, 255), -1)
    cv2.rectangle(frame, (10, 10), (330, 140), hud_color, 2)
    cv2.putText(frame, f"Score: {score}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
    cv2.putText(frame, f"Time: {remaining:05.2f}s", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
    cv2.putText(frame, f"Pinch: {pinch_text}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
    cv2.putText(frame, f"Dist: {pinch_dist:.4f}", (170, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
    cv2.putText(frame, f"FPS: {fps:05.2f}", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)


def main() -> None:
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam. Please check that a camera is connected and not in use.")
        return

    # Precompute frame size after first read to place stickman at center.
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[Error] Failed to read from webcam. Try restarting the script or your camera.")
        cap.release()
        return

    height, width = frame.shape[:2]
    center = (width // 2, height // 2)

    # Create targets
    targets = create_targets(width, height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    pinch_on = False
    hold_counter = 0
    release_counter = 0
    score = 0
    frame_count = 0
    start_time = time.time()
    fps_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Warning] Frame grab failed. Ending game loop.")
            break

        # Mirror the frame so movement feels natural.
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pinch_distance = 1.0  # Default large distance when no hand detected.
        pinch_now = False

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dx = thumb_tip.x - index_tip.x
            dy = thumb_tip.y - index_tip.y
            pinch_distance = math.hypot(dx, dy)
            pinch_now = pinch_distance < PINCH_THRESHOLD

            # Draw hand landmarks for learning/debugging.
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

        # Debounce logic: only switch states after consistent frames.
        if pinch_now:
            hold_counter += 1
            release_counter = 0
            if not pinch_on and hold_counter >= HOLD_FRAMES:
                pinch_on = True
        else:
            release_counter += 1
            hold_counter = 0
            if pinch_on and release_counter >= RELEASE_FRAMES:
                pinch_on = False

        # Update timer and compute FPS.
        elapsed = time.time() - start_time
        remaining = max(0.0, GAME_DURATION - elapsed)
        if time.time() - fps_time >= 0.5:
            fps = frame_count / (time.time() - fps_time)
            frame_count = 0
            fps_time = time.time()
        frame_count += 1

        # Draw stickman and slash (if active).
        _, right_hand = draw_stickman(frame, center)
        if pinch_on and remaining > 0:
            slash_start, slash_end = slash_line(right_hand, frame_count)
            cv2.line(frame, slash_start, slash_end, (0, 0, 255), 4)
            cv2.circle(frame, slash_end, 10, (0, 0, 200), -1)
        else:
            slash_start, slash_end = right_hand, right_hand

        # Update and draw targets. Freeze when time is up.
        for idx, target in enumerate(targets):
            if remaining > 0:
                target.move(width, height)
            cv2.circle(frame, tuple(target.position.astype(int)), TARGET_RADIUS, (0, 180, 80), -1)

        # Check hits only when slash is active and time remains.
        if pinch_on and remaining > 0:
            hits = check_hits(targets, slash_start, slash_end)
            if hits:
                score += SCORE_PER_HIT * len(hits)
                for hit_idx in hits:
                    targets[hit_idx].respawn(width, height)

        # HUD and timer end message.
        draw_hud(frame, score, remaining, pinch_on, pinch_distance, fps)
        if remaining <= 0:
            cv2.putText(frame, "TIME UP!", (width // 2 - 120, height // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Final Score: {score}", (width // 2 - 170, height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 200), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            # Reset game state
            score = 0
            start_time = time.time()
            targets = create_targets(width, height)
            pinch_on = False
            hold_counter = 0
            release_counter = 0

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
