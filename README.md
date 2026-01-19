## Hi there 👋

<!--
**sriz99/sriz99** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
# Sri Harish Madampur Suresh  
**Computer Vision & Perception Engineer | Autonomous Driving | 3D Scene Understanding**

📍 Germany | 🚗 Autonomous Driving | 🧠 Computer Vision  
🔗 [LinkedIn](https://www.linkedin.com/in/sri-harish-m-s-1772521b1) | 🔗 [GitHub](https://github.com/sriz99)

---

## 🧠 About Me

Autonomous Systems Engineer specializing in **computer vision and 3D scene understanding for autonomous driving**, based Germany. Recent **M.Eng. graduate in Automotive Systems from Hochschule Esslingen** with hands-on experience at **Daimler Truck AG (Master Thesis: Ground Truth Generation for Voxel-Based 3D Occupancy Prediction using Multi-LiDAR Truck Data)** and **Institute for Intelligent Systems**. Developed perception pipelines including VLM optimization for traffic light/sign recognition (+15% accuracy in CARLA), camera-based localization/tracking, and KITTI YOLOv8 detection system. Good Knowledge in Python, PyTorch, Numpy, Pandas, Docker, and CARLA; experience in model training, dataset curation, and geometry-aware perception for scalable AD pipelines. Committed to advancing perception algorithms, real-time autonomous systems, and robust AI solutions for intelligent mobility applications. Seeking **Computer Vision / Perception Engineer** roles to advance autonomous vehicle technologies.

---
## ⭐ Project Highlights

- 🚛 **3D Occupancy Ground Truth Pipeline** – Multi-LiDAR fusion and voxel-based label generation for autonomous trucks (−80% manual labeling effort)
- 🚦 **VLM-Enhanced Traffic Light & Sign Recognition** – +15% driving score improvement in CARLA
- 🎯 **Vision-Based Localization & Tracking** – Real-time camera-based positioning with EKF
- 🤖 **RAG-Powered QA System** – End-to-end GenAI application with LangChain and Gemini
---

## 🛠️ Featured Projects 
### 🚛 Ground Truth Generation Pipeline for 3D Occupancy Prediction in Trucks

This project involved designing and implementing an automated pipeline to generate dense 3D semantic occupancy Ground Truth labels for autonomous heavy-duty vehicles, addressing the scarcity of labeled training data.

![3D Occupancy Screenshot](assets/3D_Occupancy.png)

The  workflow included implementing multi-LiDAR sensor fusion to aggregate sparse sensor data into a unified 360-degree point cloud representation, adapting deep learning architectures to perform semantic segmentation on outdoor LiDAR scenes, and developing logic to rigorously separate static background elements from dynamic foreground objects to ensure temporal consistency and prevent motion artifacts. To tackle data sparsity, the pipeline utilized  Reconstruction methods to transform sparse point clouds into dense, watertight meshes, followed by voxelization processes to convert into discrete 3D occupancy grids. 

The project required managing complex coordinate system transformations to align multi-frame data and validating the automated output against manually annotated ground truth using Intersection-over-Union (mIoU) metrics to ensure label quality for training vision-based perception networks.


**Tech:** Python · PyTorch · multi-LiDAR fusion · Point Cloud Segmentation · Voxel Grids · Docker · Linux · Git

---

## 🚦 Vision-Language Model Optimization for Autonomous Driving  
This project focuses on enhancing the safety and compliance of Large Language Model (LLM)-based autonomous driving systems by developing a robust perception stack within the CARLA simulation environment. Addressing the critical issue of high infraction rates in end-to-end driving agents, a modular Traffic Light and Sign Recognition (TLSR) framework has been designed to integrate seamlessly with LMDrive. The system utilizes state-of-the-art YOLO11 object detection model, which has been fine-tuned specifically on CARLA datasets to ensure high-precision recognition of traffic signals and signs. To handle complex road scenarios, a 'Relevance Prediction' algorithm has been developed that dynamically identifies the specific traffic light governing the ego vehicle's lane, effectively filtering out irrelevant signals. 

<p align="center">
  <img src="assets/traffic_light.png" width="45%" />
  <img src="assets/traffic_sign.png" width="45%" />
</p>

Additionally, a 'State Validation' module was implemented using temporal consistency checks to mitigate false positives and stabilize detection outputs over consecutive frames. The final perception data is translated into natural language instructions, enabling the frozen LLM to make context-aware driving decisions without requiring extensive model retraining. Extensive evaluation using the LangAuto benchmark demonstrated that this architecture significantly reduces red light violations and improves overall driving scores compared to baseline methods. This work highlights the potential of combining traditional computer vision techniques with Generative AI to create more reliable and rule-compliant autonomous vehicles.

**Tech:** PyTorch · Vision-Language Models · CARLA · Computer Vision

---
## 🎯 Camera-Based Localization & Object Tracking

![Camera based localization Screenshot](assets/camera_based.png) 

This project implements a real-time camera-based localization and tracking system designed to function as an indoor GPS for a 1:14 scaled autonomous traffic environment,. Utilizing a ceiling-mounted IP camera, the system captures bird’s-eye view footage which is processed using MATLAB to handle image acquisition and data stream management. A critical phase of the project involved rigorous camera calibration using checkerboard patterns to correct lens distortion and accurately transform 2D pixel coordinates into 3D world coordinates. For robust object identification, the system evolved from traditional machine learning (ACF) to a high-performance Deep Learning approach using YOLOv4, optimized via transfer learning to detect vehicles with high precision. Custom ground truth datasets were meticulously prepared using MATLAB’s Video Labeler to fine-tune the detection models. To ensure smooth trajectory estimation, an Extended Kalman Filter (EKF) was integrated to track the vehicle's position, velocity, and heading angle, effectively handling non-linear motion dynamics and measurement noise,. The final system successfully demonstrates the fusion of computer vision and state estimation algorithms to provide reliable navigation data for autonomous driving research  

**Tech:** MATLAB Computer Vision and Deep Learning Toolbox · YOLO · Real-Time Vision

---

## 🚗 KITTI-Based Object Detection for Road Safety

This project demonstrates end-to-end computer vision solutions for road safety applications, implementing a complete object detection system using the KITTI Vision Benchmark Suite and YOLOv8 architecture. A full machine learning pipeline has been developed from data preparation to model deployment, leveraging Python, Streamlit, OpenCV, and Ultralytics YOLO to create an interactive web application that performs real-time inference on both images and videos. The system features a user-friendly Streamlit interface that enables users to upload media files and receive instant object detection results with bounding boxes, confidence scores, and class labels for cars, pedestrians, and cyclists . The model's performance has been validated through comprehensive evaluation using metrics like mAP, precision, and recall, ensuring robust detection capabilities across diverse road scenarios. The production-ready implementation includes proper dependency management, clear setup instructions, and deployment considerations, demonstrating capability to deliver scalable machine learning solutions. Visual results showcase the system's effectiveness with annotated outputs displaying accurate object localization and classification, making complex computer vision technology accessible and practical for real-world autonomous driving applications

**Tech:** YOLOv8 · KITTI Dataset · PyTorch · Computer Vision · Model Evaluation

---
## 🤖 RAG-Powered Webpage QA Chatbot with Gemini

![RAG Chatbot Screenshot](assets/rag_chat.png) 

This RAG-powered webpage chatbot demonstrates building end-to-end AI applications using modern technologies including Streamlit for the web interface, LangChain for RAG orchestration, Google Gemini for embeddings and LLM capabilities, and ChromaDB for efficient vector storage. The project  implements a clean three-tier architecture with modular backend design, separating concerns into specialized modules for configuration, document loading, embedding generation, and RAG chain construction . Solution to real-world challenges including webpage content extraction using WebBaseLoader, intelligent text chunking with RecursiveCharacterTextSplitter for optimal retrieval, and context-aware question answering through a sophisticated RAG pipeline that processes user queries by retrieving relevant document chunks and generating concise, accurate responses. The Streamlit based web application features a professional chat-style interface with conversation history management, proper session state handling, and comprehensive deployment documentation that demonstrates production-ready development practices

**Tech:** LLMs · RAG · LangChain · Google Gemini · ChromaDB · Streamlit · Vector Databases · Prompt Engineering

---

## 🧰 Skills

**Programming & Machine Learning**  
Python · PyTorch · NumPy · Pandas · OpenCV · MATLAB  

**Computer Vision & 3D Perception**  
Object Detection · Point Cloud Processing · 3D Scene Understanding · Occupancy Networks · Multi-LiDAR Fusion · Semantic Segmentation  

**State Estimation & Geometry**  
Coordinate Transformations · Sensor Calibration · Voxelization · Geometry-Aware Perception  

**Simulation & Robotics**  
CARLA · ROS · Simulation-Based Evaluation  

**Tools & Platforms**  
Linux · Git · Docker · Streamlit · ChromaDB

---

## 🎯 Research Interests

- 3D Occupancy Prediction  
- Vision-Language Models for Autonomous Driving  
- Multi-Sensor Fusion  
- Simulation-Based Evaluation  

---

📫 **Contact:** sriharish52@outlook.com
