# 🤖 Face Recognition Web App

This is a simple Face Recognition web application built using OpenCV, Streamlit, and Python. It allows you to recognize faces either from a static image or a webcam feed.

---

## 🚀 Features

- Face detection using Haar Cascade Classifier.
- Face recognition using OpenCV’s LBPH (Local Binary Patterns Histograms) algorithm.
- Interactive user interface via Streamlit.
- Two modes: **Static Image Upload** and **Live Webcam Recognition**.
- Displays predicted name and confidence level.
- Labels unrecognized faces as **"Unknown"**.

---

## 🧠 How It Works

This app uses OpenCV's **LBPH face recognizer**, which functions similarly to modern face recognition models it just sort of training deeplearning model but not from stratch but trained predefined opencv model.

Instead, it relies on OpenCV’s built-in algorithms for feature extraction and matching, which work well for small-scale applications.


⚠️ Limitations:
This prototype was trained on a very limited dataset, with only a few images per class.
As a result, the recognition accuracy is not optimal.
Given more time and resources, a larger and more diverse dataset can be used to significantly improve performance.
---
