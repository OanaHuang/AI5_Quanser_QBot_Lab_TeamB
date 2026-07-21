# AI5 Quanser QBot Lab — Team B

This repository contains the project work completed by **Team B** for the **Advanced AI 5** course at the **University of Glasgow**.

The project was developed by students from the **Robotics & AI** programme using the **Quanser QBot platform**.

The main objective was to develop a CNN-based control system that uses images from the QBot's downward-facing camera to recognise different track situations and select suitable driving parameters, including steering behaviour, PID gains, and forward velocity.

The original DIP-based control files are also retained for reference and comparison.

---

## Project Overview

The project was completed in the following stages:

1. Develop and tune the original line-following controller.
2. Test suitable PID gains and velocities for different track conditions.
3. Collect and label images from the QBot's downward-facing camera.
4. Train a CNN model to classify track situations.
5. Use the CNN prediction to adjust the QBot's driving behaviour.
6. Evaluate the trained model and compare different control approaches.

The track conditions considered during development included:

- Straight sections
- Curved sections
- Crossroads
- Different approach directions
- Different robot positions relative to the track centre

The early project planning also considered using different velocity and PID settings for straight lines, corners, and crossroads. :contentReference[oaicite:0]{index=0}

---

## Repository Structure

```text
AI5_Quanser_QBot_Lab_TeamB/
├── Base_Cnn_Model/
│   ├── baseline_cnn_checkpoint.pth
│   ├── evaluation1.jpeg
│   ├── evaluation2.jpeg
│   └── line_following_checkpoint_version.py
│
├── Dataset_sumup/
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
