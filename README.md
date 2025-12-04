# **TransAID**

## **Overview**
[![TransAID Demo Video](https://img.youtube.com/vi/WJ9eIU-YcbY/maxresdefault.jpg)](https://youtu.be/WJ9eIU-YcbY)
**TransAID** is a diagnostic-assistance tool integrated with custom software to detect secondary caries using a YOLOv8-based computer vision model.

## **Key Features**
- **10× more cost-efficient** than commercial devices, with a compact physical design.  
- Eliminates radiation exposure by using **Near-Infrared Light Transillumination (NILT)** at **980 nm**.  
- **Photogrammetry-based 3D reconstruction** of tooth surfaces.  
- Two functions in one device:  
  - Secondary caries detection  
  - 3D tooth reconstruction for visualization  
- YOLOv8 performance:  
  - **Accuracy:** 90%  
  - **Sensitivity:** 92.9%  
  - **Specificity:** 87.5%  
  - **Precision:** 86.7%  
  - **False Positive Rate:** 12.5%

## **System Architecture**
![System Architecture](https://github.com/stnrliza/transaid/blob/master/readme-images/architecture.png)

## **3D Design** 
![3D Design](https://github.com/stnrliza/transaid/blob/master/readme-images/3d.gif)


## **Software Flow and Description**
![Software Flow](https://github.com/stnrliza/transaid/blob/master/readme-images/software-flow.png)

### **Software Files**
| File Name | Description |
|----------|-------------|
| **a_welcome_screen.py** | Landing page where users choose between starting a new examination or viewing history. |
| **b1_patient_data.py** | Creates a new patient folder and stores patient name & examination date in the local SQLite database (`pasien.db`). |
| **b2_diagnosis_history.py** | Displays patient history: name, exam date, and caries predictions. |
| **c_live_camera.py** | Shows live camera feed and captures images via `start_push_button.py`, storing them into the database. |
| **d_loading_screen.py** | Displays a loading bar while waiting for YOLOv8 processing. |
| **e_diagnosis_result.py** | Displays the secondary caries prediction results. |
| **main.py** | Main entry point coordinating all screens and software logic. |
| **pasien.db** | Local patient database using SQLite3. |

### **Computer Vision (YOLOv8) Files**
| File Name | Description |
|----------|-------------|
| **yolov8_segment.py** | Post-processing pipeline loading the YOLOv8 model (`best.pt`).<br>Implements:<br>• Green overlay for confidence **1–49%**<br>• Red overlay for **50–100%**<br>• Confidence text rendering<br>Outputs are shown in `e_diagnosis_result.py` and saved into `pasien.db`. |
| **best.pt** | The best-performing YOLOv8 model obtained from training. |

### **Device Logic Files**
| File Name | Description |
|----------|-------------|
| **start_push_button.py** | Reads Arduino serial input (`Program_Fix.ino`) to detect button presses. |
| **Program_Fix.ino** | Arduino program detecting push-button state (pressed / not pressed). |

## **User Flow**
> This device must be operated under the supervision of a qualified operator.  
> It functions as a **decision-support tool**, not a standalone diagnostic instrument.
1. Operator and patient wear infrared-filter safety glasses.  
2. Operator positions the device inside the oral cavity in a darkened environment.  
3. Operator presses the device’s push button to capture an image.  
4. Software displays the predicted secondary caries segmentation and analysis.

## **YOLOv8 Model**
A custom-trained YOLOv8 model is used for secondary caries detection and segmentation.  
The model is trained on a curated infrared dental dataset.

### **Training Configuration**
- **Base model:** `yolov8l.pt`  
- **Image size:** 640×640  
- **Epochs:** 500  
- **Optimizer:** AdamW  
- **Learning rate:** 0.002 (cosine LR scheduling)  
- **Batch size:** Auto (`-1`)  
- **Device:** GPU (fallback to CPU)  
- **Validation:** Every epoch  
- **Mixed Precision (AMP):** Enabled  
- **Early stopping:** Patience 50  

### **Data Augmentation**
- Vertical flip: 0.5  
- Horizontal flip: 0.5  
- Rotation: ±15°  
- Translation: 0.1  
- Scaling: 0.2  
- Shear: 10  
- Mosaic: 0.8  
- Mixup: 0.2  

### **Dataset Configuration**
Defined in `data.yaml`:
- Train & validation image paths  
- Class labels  
- YOLO segmentation annotation format  

### **Training Script Summary**
1. Detect GPU availability  
2. Load pretrained YOLOv8 model  
3. Train with dataset & augmentations  
4. Save logs and checkpoints  
5. Export **best.pt** based on validation performance  
Final model weight path:
<TrainingFolder>/train_best_l_box/weights/best.pt

## **Model Output**
The YOLOv8 model performs:
- Secondary caries region prediction  
- Segmentation mask generation  
- Caries percentage calculation  
- Visual overlays with confidence values  
This AI model powers the diagnostic assistance capabilities of TransAID.

## **Team & Contributions**
- **Inggil Ma’rifat Djati** — Team Lead, Project Management, Scientific Writing
- **Farouq Akbar Aldy** — Electrical Wiring, 3D Design, Dataset Labeling  
- **Siti Nurhaliza (me)** — Software Development, Machine Learning Deployment, Scientific Writing  
- **Zhafira Alya Afanin** — Literature Review, Scientific Writing  
- **Childnandira Ayu Nur Ittazza** — Literature Review, Dataset Labeling  

## **Acknowledgements**
We express our sincere appreciation to **Universitas Brawijaya** for providing the resources and support that enabled this project.  
We also extend our gratitude to our supervising lecturer, **Eka Maulana, S.T., M.T., M.Eng.**, for his guidance, technical direction, and continuous mentorship throughout the research and development process.
