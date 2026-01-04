# Pinch Ninja Stickman / ピンチ・ニンジャ・スティックマン

## What is this? これは何？
- **EN:** A webcam mini-game where you pinch your fingers to swing a stickman’s sword and pop bouncing targets. Runs with only OpenCV + MediaPipe + NumPy.
- **JP:** ウェブカメラで指をつまむと棒人間が剣を振り、跳ねる的を壊すミニゲームです。OpenCV・MediaPipe・NumPyだけで動きます。

## Quickstart (copy & paste)
1. Clone the project  
   ```bash
   git clone https://github.com/hirasunaryou/pinch-ninja-stickman.git
   cd pinch-ninja-stickman
   ```
2. Create a virtual environment  
   - **Windows (PowerShell):**  
     ```powershell
     python -m venv .venv
     .\\.venv\\Scripts\\Activate.ps1
     ```
   - **macOS / Linux (Terminal):**  
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
4. Run the game  
   ```bash
   python pinch_ninja_stickman.py
   ```

## Controls
- Press **q** → quit the game.
- Press **r** → restart with fresh targets and timer.

## EDIT HERE (tune the game feel)
Open `pinch_ninja_stickman.py` and find the top section marked `EDIT HERE`. These are the safe knobs to tweak:
- `PINCH_THRESHOLD` (pinch sensitivity)  
- `HOLD_FRAMES` / `RELEASE_FRAMES` (debounce for pinch on/off)  
- `TARGET_COUNT`, `TARGET_RADIUS`, `TARGET_SPEED_RANGE` (how many targets, size, speed)  
- `SLASH_LENGTH`, `SLASH_SWEEP_DEG`, `HIT_RADIUS` (sword reach and hit detection)  
- `SCORE_PER_HIT`, `GAME_DURATION`, `WINDOW_NAME` (points, round length, window title)

## Troubleshooting
- **Camera permission is blocked**  
  - Windows: allow camera access in *Settings → Privacy & security → Camera*.  
  - macOS: allow in *System Settings → Privacy & Security → Camera* for your terminal/Python.  
  - Linux: ensure your user can access the webcam device (e.g., add to `video` group) and try restarting the terminal.
- **Camera is already in use**  
  - Close other apps (Teams/Zoom/browser tabs) that might hold the webcam.  
  - Unplug/replug the webcam or switch to another camera index in `cv2.VideoCapture(0)` if needed.

## Developer: postcall trace bundles / 開発者向けトレースバンドル
- **EN:** The file `postcall_trace.py` writes a redacted debug bundle to `.dal_logs/postcall/runs/YYYYMMDD/<requestId>/` so you can trace how an LLM pipeline behaved without storing raw prompts or transcripts. Run it directly to generate a safe demo bundle:
  ```bash
  python postcall_trace.py
  ```
- **JP:** `postcall_trace.py` は `.dal_logs/postcall/runs/YYYYMMDD/<requestId>/` にマスク済みのトレースバンドルを出力します。生のプロンプトや会話を保存しない設計なので安心です。デモを作るには次を実行してください:
  ```bash
  python postcall_trace.py
  ```

## Lessons / ミッション集
See **LESSONS.md** for beginner-friendly missions you can try while playing. Screenshot ideas and small effects are included.
