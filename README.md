#  Hand Gesture Puzzle

A real-time computer vision-based interactive puzzle game where users create and solve image puzzles using **hand gestures instead of a mouse or touch input**.

Built using **OpenCV** and **MediaPipe**, this project tracks hand movements in real-time and enables a fully gesture-controlled experience.

---

## 🎬 Demo

▶️ [Watch Demo Video](https://drive.google.com/file/d/1zRgMhcaL9ZyqrnTejsljVON1PPJveMRH/view?usp=drive_link)

---

## ✨ Features

* 🖐️ Real-time hand tracking using MediaPipe
* ✌️ Two-hand frame selection (set your puzzle area)
* 🤏 Pinch gesture to capture image
* 🧩 Automatic puzzle generation (3x3 grid)
* 🎯 Drag-and-drop puzzle solving using gestures
* 🧠 Smooth pointer tracking
* ⏱️ Timer-based gameplay
* 🔀 Shuffle animation before solving
* 🎮 Fully interactive without keyboard/mouse

---

## 🛠 Tech Stack

* **Python**
* **OpenCV** – for video processing & rendering
* **MediaPipe** – for hand tracking & gesture detection

---

## ⚙️ How to Run

```bash
git clone https://github.com/molly22-byte/hand-gesture-puzzle.git
cd hand-gesture-puzzle
pip install -r requirements.txt
python main.py
```

---

## 🧠 How It Works

1. Camera captures live video using OpenCV
2. MediaPipe detects hand landmarks in real-time
3. User selects a frame using both hands
4. Pinch gesture captures the selected frame
5. Image is split into puzzle tiles
6. User solves puzzle by dragging tiles using pinch gestures

---

## 🚀 Future Improvements

* 🎮 Difficulty levels (4x4, 5x5 puzzles)
* 🌐 Web-based version (no install required)
* 👥 Multiplayer / competition mode
* 📱 Mobile compatibility
* 🧠 AI-based gesture improvements

---