# Face-Recognition
# 🧠 Face Recognition System using ArcFace & SVM

A high-accuracy, real-time face recognition system using state-of-the-art deep learning models and machine learning classifiers. This project compares multiple face recognition architectures and implements a robust real-time recognition prototype using the best-performing model.

## 📌 Problem Statement

Automated identity verification is essential for security, access control, and digital organization. However, face recognition faces several challenges:
- Accurately recognizing faces under varying lighting, pose, and expression.
- Identifying unknown individuals not present in the training set.
- Selecting optimal models for both high accuracy and performance.

## ✅ Proposed Solution

This project presents an end-to-end face recognition pipeline:
- **Face Detection & Embedding**: 
  - 📌 ArcFace (InsightFace): High-accuracy embeddings.
  - 📌 FaceNet (comparative analysis only).
- **Classification**:
  - Used ML classifiers like SVM, KNN, MLP, LDA.
  - Best accuracy achieved using **LDA** with ArcFace.
- **Unknown Face Detection**:
  - Implemented using **One-Class SVM**.
- **Real-Time Recognition**:
  - Webcam-based prototype developed using Python and OpenCV.

## 🧪 System Overview

- **Language**: Python 3.x  
- **Environment**: Jupyter Notebook (Google Colab)  
- **Libraries**:  
  - `opencv-python`, `insightface`, `onnxruntime`, `scikit-learn`, `numpy`, `matplotlib`, `torch`  
  - `facenet-pytorch` for comparison  
- **Models Used**:  
  - ArcFace (InsightFace) for face embeddings  
  - LDA and SVM for classification  
  - One-Class SVM for anomaly (unknown) detection  

## 🛠️ Features

- ✅ Multiple face detection and recognition in one frame.
- ✅ Unknown face classification using One-Class SVM.
- ✅ Real-time webcam recognition prototype (locally runnable).
- ✅ Clear performance comparison between ArcFace and FaceNet.
- ✅ Clean UI output using `cv2_imshow` in Colab.

## 🚀 Getting Started

### Installation
Clone this repository:
```bash
git clone https://github.com/naresh-github-2005/face-recognition-project.git
cd face-recognition-project
````

Install required libraries:

```bash
pip install opencv-python insightface onnxruntime scikit-learn facenet-pytorch
```

### Usage

1. Train face embeddings from known images.
2. Load models (`svm_model.pkl`, `ocsvm_model.pkl`, etc.).
3. Run the recognition notebook.
4. Test on new images or webcam.

## 📊 Results

| Model   | Classifier | Accuracy (%) |
| ------- | ---------- | ------------ |
| ArcFace | LDA        | **98.0**     |
| ArcFace | SVM        | 96.3         |
| FaceNet | SVM        | 89.7         |

### Sample Output

* ✅ Known face detection
* 🚫 Unknown face flagged by One-Class SVM
* 🎥 Real-time webcam frame processing

Sample Output 1:
![image](https://github.com/user-attachments/assets/ae501125-d2fd-4a0f-9679-81a8d024c0b8)

Sample Output 2:
![image](https://github.com/user-attachments/assets/f4fda0b7-3b3b-4a07-a0bf-26ef7f9fb747)



## 🧠 Future Scope

* 🔎 **Vector Database Integration** (e.g., FAISS) for large-scale identity management.
* 🌐 **Web API Deployment** using FastAPI/Flask + Docker.
* 📈 **Bias & Fairness Auditing** for better generalization.
* 🧪 **Fine-tuning ArcFace** on custom datasets.

## 📚 References

* **InsightFace**: [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
* **FaceNet**: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
* **Dataset**: [Face Recognition Dataset - Kaggle](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset)

## 👤 Author

**Naresh R**
🎓 VIT Vellore | Software Engineering
📧 [naresh.r2022a@vitstudent.ac.in](mailto:naresh.r2022a@vitstudent.ac.in)


> ⭐ Star this repo if you find it useful
> 🤝 Contributions and feedback welcome!
