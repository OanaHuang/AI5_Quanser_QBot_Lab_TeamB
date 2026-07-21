# CNN-Adaptive Line Following with Virtual QBot Platform

This repository contains the work completed by **Team B** for the **ENG5337 Advanced Artificial Intelligence & Machine Learning** course at the **University of Glasgow**.

The project explores **scene-aware adaptive line following** using the **Quanser Virtual QBot Platform in QLabs**. A custom convolutional neural network classifies the current road scene, and the predicted scene is then used to adjust the behaviour of a low-level line-following controller.

The project has also been featured in **Quanser's Community Showcase**.

---

## Project Overview

Traditional line-following systems often use a fixed set of control parameters for all road conditions. However, the most suitable velocity, steering behaviour, and PID gains may vary depending on whether the robot is travelling on a straight section, approaching a curve, or entering a junction.

This project combines:

- Deep learning
- Classical PID-style control
- Robot simulation
- Scene classification
- Adaptive control parameter selection

The CNN acts as a high-level perception layer. It does not replace the low-level controller. Instead, it identifies the current road condition and selects more suitable control parameters for that scene.

A simplified workflow is shown below:

```text
Downward-Facing Camera
          ↓
   Image Preprocessing
          ↓
      CNN Classifier
          ↓
 Road Scene Prediction
          ↓
Control Parameter Selection
          ↓
Line-Following Controller
          ↓
      QBot Movement
