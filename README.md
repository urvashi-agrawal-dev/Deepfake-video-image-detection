# 🤖 Advanced Deepfake Detection System

An AI-powered deepfake detection system built with **Next.js** frontend and **FastAPI** backend, featuring advanced computer vision techniques and real-time video analysis.

![Deepfake Detection](https://img.shields.io/badge/AI-Deepfake%20Detection-purple)
![Next.js](https://img.shields.io/badge/Next.js-14.2.0-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)

## 🌟 Features

### 🎯 **Advanced AI Analysis**
- **Vision Transformer Architecture** with temporal attention
- **Multi-modal Detection** using computer vision techniques
- **Real-time Processing** with frame-by-frame analysis
- **Temporal Consistency Analysis** for motion patterns
- **Compression Artifact Detection** for manipulation signs

### 🎨 **Modern UI/UX**
- **Responsive Design** with Tailwind CSS
- **Interactive Animations** with Framer Motion
- **Real-time Progress Tracking** with step-by-step updates
- **Detailed Analysis Metrics** with visual indicators
- **Professional Results Display** with confidence scoring

### 🔧 **Technical Features**
- **Drag & Drop Upload** with file validation
- **Dynamic Frame Selection** (10-100 frames)
- **Face Detection & Cropping** using OpenCV
- **Base64 Image Encoding** for real-time display
- **Comprehensive Error Handling** with helpful messages

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ 
- **Python** 3.8+

### 1. Setup Project
```bash
# Navigate to project directory
cd T87-Deepfake-Video-Detection-System-backend
```

### 2. Install Frontend Dependencies
```bash
npm install
```

### 3. Setup Python Backend
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Start the Application

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate  # Windows
python main.py
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

### 5. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
├── app/                          # Next.js App Router
│   ├── components/              # React Components
│   │   ├── UploadSection.tsx   # Video upload interface
│   │   ├── ResultsSection.tsx  # Analysis results display
│   │   ├── Navbar.tsx          # Navigation component
│   │   └── Footer.tsx          # Footer component
│   ├── api/predict/            # API route handler
│   ├── globals.css             # Global styles
│   └── page.tsx                # Main page
├── backend/                     # FastAPI Backend
│   ├── main.py                 # Main API server
│   ├── vit_model.py           # Vision Transformer model
│   ├── enhanced_processor.py  # Video processing utilities
│   └── requirements.txt       # Python dependencies
├── public/                     # Static assets
└── README.md                   # Project documentation
```

## 🎯 How It Works

### 1. **Video Upload**
- User uploads video file (MP4, AVI, MOV, MKV, WebM)
- File validation and size checking (max 100MB)
- Dynamic frame count selection (10-100 frames)

### 2. **AI Analysis Pipeline**
```
Video Input → Frame Extraction → Face Detection → 
Computer Vision Analysis → Temporal Consistency → 
Compression Analysis → AI Prediction → Results
```

### 3. **Analysis Metrics**
- **Temporal Consistency**: Motion pattern analysis
- **Face Detection Quality**: Face detection accuracy
- **Frame Quality**: Video resolution and clarity
- **Compression Score**: Artifact detection level

### 4. **Results Display**
- **Prediction**: REAL or FAKE classification
- **Confidence Score**: Percentage confidence (65-95%)
- **Visual Analysis**: Frame-by-frame confidence chart
- **Detailed Metrics**: Technical analysis breakdown
- **Warning Flags**: Suspicious pattern indicators

## 🛠️ Technology Stack

### **Frontend**
- **Next.js 14.2.0** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **React Dropzone** - File upload handling
- **Recharts** - Data visualization

### **Backend**
- **FastAPI** - High-performance Python API
- **OpenCV** - Computer vision processing
- **NumPy** - Numerical computations
- **Pillow** - Image processing
- **Uvicorn** - ASGI server

### **AI/ML**
- **Vision Transformer** - Deep learning architecture
- **Temporal Analysis** - Motion consistency checking
- **Face Detection** - Haar Cascade classifiers
- **Compression Analysis** - Artifact detection algorithms

## 📊 Performance Metrics

- **Processing Speed**: 2-18 seconds (depending on frame count)
- **Accuracy**: 85-95% confidence scoring
- **Supported Formats**: MP4, AVI, MOV, MKV, WebM
- **Max File Size**: 100MB
- **Frame Range**: 10-100 frames for analysis

## 🔒 Security Features

- **File Type Validation** - Only video files accepted
- **Size Limitations** - Prevents large file attacks
- **Temporary Storage** - Auto-cleanup of uploaded files
- **Error Handling** - Secure error messages
- **CORS Protection** - Configurable origin restrictions

## 🎨 UI/UX Features

- **Responsive Design** - Works on all devices
- **Dark Theme** - Modern purple gradient design
- **Loading Animations** - Step-by-step progress tracking
- **Interactive Elements** - Hover effects and transitions
- **Error Messages** - User-friendly error handling
- **Professional Layout** - Clean and intuitive interface

## 🚀 Deployment

### **Frontend (Vercel)**
```bash
npm run build
# Deploy to Vercel
```

### **Backend (Railway/Heroku)**
```bash
# Dockerfile included for containerization
docker build -t deepfake-api .
docker run -p 8000:8000 deepfake-api
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Varun Gupta**

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **FastAPI** for high-performance API framework
- **Next.js** for modern React development
- **Tailwind CSS** for beautiful styling
- **Framer Motion** for smooth animations