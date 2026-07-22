# CNN-Adaptive Line Following with Virtual QBot Platform

We are **Team B** from the **ENG5337 Advanced Artificial Intelligence & Machine Learning** course at the **University of Glasgow**.

In this project, we explored **scene-aware adaptive line following** using the **Quanser Virtual QBot Platform in QLabs**. We developed a custom convolutional neural network to classify the current road scene. We then used the predicted scene to adjust the behaviour of a low-level line-following controller.


---

## Project Overview

Traditional line-following systems often use a fixed set of control parameters for all road conditions. However, we found that the most suitable velocity, steering behaviour, and PID gains may vary depending on whether the robot is travelling on a straight section, approaching a curve, or entering a junction.

In this project, we combined:

- Deep learning
- Classical PID-style control
- Robot simulation
- Scene classification
- Adaptive control parameter selection

We used the CNN as a high-level perception layer. It does not replace the low-level controller. Instead, it identifies the current road condition and helps us select more suitable control parameters for that scene.

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

AI5_Quanser_QBot_Lab_TeamB/
├── Base_Cnn_Model/
│   ├── baseline_cnn_checkpoint.pth
│   ├── evaluation1.jpeg
│   ├── evaluation2.jpeg
│   └── line_following_checkpoint_version.py
│
├── Dataset_sumup/
│   ├── dataset_collection/
│   ├── dataset_cross/
│   └── dataset_speedlabel/
│
├── Final Materials/
│   ├── baseline_cnn_checkpoint.pth
│   ├── CNN/
│   ├── Quanser/
│   └── cnn_architecture_editable.svg
│
├── Final Report/
│   ├── AI5_B_presentation.pptx
│   └── train.py
│
├── Notes/
│
├── TestScripts/
│   ├── deploy_cnn_test_v1.0.py
│   ├── deploy_cnn_test_v1.1.py
│   ├── evaluate_csv_test_v1.0.py
│   └── line_following_cnn.py
│
├── flow chart.vsdx
└── README.md
