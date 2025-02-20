# HeadHunter 🎯

A computer vision-based aim assistance tool running at 16-24 FPS for counter strike using OpenCV template matching.



## 🧠 Core Features

- Real-time head detection using template matching algorithms
- Optimized performance achieving 16-24 FPS on mid-range hardware
- Low-latency mouse movement simulation

## ⚙️ Technical Implementation

HeadHunter utilizes OpenCV's template matching algorithm to detect potential targets:

1. Capture screen region using `mss`
2. Preprocess image (grayscale conversion, noise reduction)
3. Apply template matching with `cv2.matchTemplate()`
4. Filter results based on confidence threshold
5. Calculate target coordinates
6. Simulate mouse movement using `pywin32`

This process repeats in real-time, providing continuous aim assistance.

## 📊 Performance Tips

- Use a GPU-enabled build of OpenCV for significant speed improvements
- Reduce `SCREEN_REGION` size to focus on the most relevant area
- Experiment with `TEMPLATE_SCALING` to balance accuracy and speed
- Close unnecessary background applications to free up system resources
- Consider overclocking your CPU/GPU for additional performance gains

---
