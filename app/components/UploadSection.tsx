'use client'

import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Upload, XCircle, Loader2, ArrowLeft } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { toast } from 'react-hot-toast'
import { DetectionResult } from '../page'

interface UploadSectionProps {
  onResult: (result: DetectionResult) => void
  onBack: () => void
}

export default function UploadSection({ onResult, onBack }: UploadSectionProps) {
  const [file, setFile] = useState<File | null>(null)
  const [sequenceLength, setSequenceLength] = useState(40)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      
      // Validate file size (100MB max)
      if (file.size > 100 * 1024 * 1024) {
        toast.error('File size must be less than 100MB')
        return
      }
      
      setFile(file)
      toast.success('Video uploaded successfully!')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1,
    multiple: false,
  })

  const handleUpload = async () => {
    if (!file) {
      toast.error('Please select a video file first')
      return
    }

    setIsProcessing(true)
    setProgress(0)

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return 90
        }
        return prev + 10
      })
    }, 500)

    const formData = new FormData()
    formData.append('upload_video_file', file)
    formData.append('num_frames', sequenceLength.toString())

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        let errorMessage = 'Processing failed'
        try {
          const error = await response.json()
          errorMessage = error.detail || error.message || errorMessage
        } catch {
          // If response is not JSON, get text
          const text = await response.text()
          errorMessage = text || `Server error: ${response.status}`
        }
        throw new Error(errorMessage)
      }

      const data = await response.json()
      
      clearInterval(progressInterval)
      setProgress(100)
      
      setTimeout(() => {
        onResult(data)
        toast.success('Video analyzed successfully!')
      }, 500)
      
    } catch (error) {
      clearInterval(progressInterval)
      console.error('Upload error:', error)
      
      let errorMessage = 'Failed to process video. Please try again.'
      
      if (error instanceof Error) {
        errorMessage = error.message
        
        // Add helpful hints for common errors
        if (error.message.includes('Cannot connect') || error.message.includes('fetch')) {
          errorMessage = '🔌 Cannot connect to AI server\n\n💡 Solutions:\n• Check if backend is running\n• Try refreshing the page\n• Contact support if issue persists'
        } else if (error.message.includes('File must be a video')) {
          errorMessage = '📹 Invalid file format\n\n💡 Please upload:\n• MP4, AVI, MOV, MKV, or WebM files\n• Maximum size: 100MB'
        } else if (error.message.includes('frames must be between')) {
          errorMessage = '⚙️ Invalid frame count\n\n💡 Please select:\n• Between 10-100 frames\n• Use slider to adjust'
        }
      }
      
      toast.error(errorMessage, { duration: 6000 })
    } finally {
      setIsProcessing(false)
      setProgress(0)
    }
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto"
      >
        {/* Back Button */}
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-purple-200 hover:text-white mb-8 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Home
        </button>

        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Upload Video for Analysis</span>
          </h1>
          <p className="text-xl text-purple-200">
            Upload a video file to detect if it's authentic or manipulated
          </p>
        </div>

        {/* Upload Card */}
        <div className="glass-effect p-8 glow-effect">
          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`upload-zone cursor-pointer ${
              isDragActive ? 'border-purple-400 bg-purple-400/10' : ''
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="w-16 h-16 text-purple-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">
              {isDragActive ? 'Drop your video here' : 'Drag & Drop Video'}
            </h3>
            <p className="text-purple-200 mb-4">
              or click to browse files
            </p>
            <p className="text-sm text-purple-300">
              Supported: MP4, AVI, MOV, MKV, WebM (Max 100MB)
            </p>
          </div>

          {/* File Preview */}
          {file && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8"
            >
              <div className="glass-effect p-6">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <p className="text-white font-medium text-lg">{file.name}</p>
                    <p className="text-purple-200 text-sm mt-1">
                      Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={() => setFile(null)}
                    className="text-red-400 hover:text-red-300 transition-colors"
                    disabled={isProcessing}
                  >
                    <XCircle className="w-6 h-6" />
                  </button>
                </div>

                {/* Sequence Length Selector */}
                <div className="mb-6">
                  <label className="block text-white font-medium mb-3">
                    Analysis Depth: <span className="text-purple-400">{sequenceLength} frames</span>
                    <span className="text-sm text-purple-300 ml-2">
                      ({sequenceLength <= 30 ? 'Fast' : sequenceLength <= 60 ? 'Balanced' : 'Thorough'})
                    </span>
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="10"
                    value={sequenceLength}
                    onChange={(e) => setSequenceLength(Number(e.target.value))}
                    disabled={isProcessing}
                    className="w-full h-2 bg-purple-900/50 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <div className="flex justify-between text-xs text-purple-300 mt-2">
                    <span>10 (2s)</span>
                    <span>30 (5s)</span>
                    <span>50 (8s)</span>
                    <span>70 (12s)</span>
                    <span>100 (18s)</span>
                  </div>
                  <p className="text-sm text-purple-300 mt-2">
                    💡 <strong>Tip:</strong> More frames = higher accuracy but longer processing time
                  </p>
                </div>

                {/* Progress Bar */}
                {isProcessing && (
                  <div className="mb-6">
                    <div className="flex justify-between text-sm text-purple-200 mb-2">
                      <span>
                        {progress < 20 ? 'Uploading video...' :
                         progress < 40 ? 'Extracting frames...' :
                         progress < 60 ? 'Detecting faces...' :
                         progress < 80 ? 'Analyzing patterns...' :
                         'Generating results...'}
                      </span>
                      <span>{progress}%</span>
                    </div>
                    <div className="w-full h-3 bg-purple-900/50 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                        className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600 relative"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
                      </motion.div>
                    </div>
                    <div className="text-xs text-purple-300 mt-2 text-center">
                      🤖 AI is analyzing your video with advanced computer vision...
                    </div>
                  </div>
                )}

                {/* Analyze Button */}
                <button
                  onClick={handleUpload}
                  disabled={isProcessing}
                  className="w-full btn-primary flex items-center justify-center gap-2 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing Video...
                    </>
                  ) : (
                    'Analyze Video'
                  )}
                </button>
              </div>
            </motion.div>
          )}
        </div>

        {/* Info Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-8 glass-effect p-6"
        >
          <h3 className="text-lg font-semibold mb-4">What happens next?</h3>
          <ul className="space-y-3 text-purple-200">
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">1.</span>
              <span>Your video is securely uploaded and processed on our servers</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">2.</span>
              <span>AI extracts frames and detects faces using advanced computer vision</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">3.</span>
              <span>Deep learning model analyzes temporal patterns and facial features</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">4.</span>
              <span>You receive a detailed report with confidence scores and visualizations</span>
            </li>
          </ul>
        </motion.div>
      </motion.div>
    </div>
  )
}
