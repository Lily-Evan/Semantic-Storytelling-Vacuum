# 🧹 Semantic Storytelling Vacuum

A research-grade prototype of an **autonomous robot vacuum** that not only maps rooms and avoids obstacles, but also:
- Builds **semantic understanding** of the house (rooms, objects).
- Learns **temporal patterns of dirt accumulation** per room.
- Plans **proactive cleaning** (e.g., Kitchen before dinner, Living Room in the morning).
- Explains its decisions with **natural-language rationales** (Explainable AI).

---

## ✨ Features
- **Apartment world generator** with rooms, doors, walls, and furniture.
- **Semantic perception (mock)**: assigns each room a tag (LivingRoom, Kitchen, Bedroom, Study).
- **Temporal dirt model** using Exponential Moving Average (EMA) + priors + simulated seasonality.
- **Planner**:
  - A* for transit.
  - Boustrophedon sweep for coverage inside rooms.
- **Explainability**: generates short Greek/English explanations for why each room is prioritized.
- **Battery simulation**: returns to dock when low.
- **Visualization**: live OpenCV window + annotated HUD (battery, coverage %, predictions).
- **Video export**: saves `semantic_vacuum.mp4` showing the full run.

---

## 🛠️ Requirements
- Python 3.8+
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/) (`opencv-python`)

Install dependencies:
```bash
pip install numpy opencv-python
▶️ Run
bash

python semantic_vacuum.py
Press ESC to stop simulation early.
At the end, you’ll see logs with final coverage, steps, and battery, and a saved video:

makefile

Finished. Coverage=87.32%  Steps=1023  Battery=177.
Video: semantic_vacuum.mp4
📂 Project Structure
bash

vacuum.py   # Main simulation code
README.md            # This file
semantic_vacuum.mp4  # Output video (generated after run)
🚀 Extensions / Research Ideas
Integrate a CNN classifier (Places365) for real room-type recognition.

Add dynamic obstacles (pets, humans) → trigger replanning.

Multi-criteria optimization: coverage vs. energy vs. rotations.

Reinforcement Learning for adaptive cleaning policies.

Human-in-the-loop explanations (chat with your vacuum).

