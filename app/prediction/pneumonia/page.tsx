'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Wind, TrendingUp, AlertCircle, Upload, X, Eye } from 'lucide-react';
import { useAuthStore } from '@/lib/store/authStore';
import Navigation from '@/components/ui/Navigation';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import Image from 'next/image';

export default function PneumoniaPredictionPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();
  const [loading, setLoading]             = useState(false);
  const [prediction, setPrediction]       = useState<any>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview]   = useState<string | null>(null);
  const [gradcamImage, setGradcamImage]   = useState<string | null>(null);
  const [showGradcam, setShowGradcam]     = useState(false);

  useEffect(() => {
    if (!isAuthenticated) router.push('/signin');
  }, [isAuthenticated, router]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target?.result as string);
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    maxFiles: 1,
    multiple: false,
  });

  const handlePredict = async () => {
    if (!selectedImage) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      const response = await fetch('/api/predict/pneumonia', { method: 'POST', body: formData });
      const result = await response.json();
      setPrediction(result);
      if (result.gradcam_image) { setGradcamImage(result.gradcam_image); setShowGradcam(true); }
    } catch (err) { console.error('Prediction failed:', err); }
    finally { setLoading(false); }
  };

  const handleReset = () => {
    setPrediction(null); setSelectedImage(null); setImagePreview(null);
    setGradcamImage(null); setShowGradcam(false);
  };

  if (!isAuthenticated) return null;
  const isPneumonia = prediction?.prediction === 'Pneumonia';

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-red-50 to-gray-100 dark:from-gray-950 dark:via-red-950/20 dark:to-gray-900">
      <Navigation />
      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">

          {/* Header */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
            <button onClick={() => router.push('/home')} className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 mb-4">
              <ArrowLeft className="w-4 h-4" /> Back to Home
            </button>
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-indigo-500 to-blue-500 flex items-center justify-center">
                <Wind className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Pneumonia Detection</h1>
                <p className="text-gray-600 dark:text-gray-400">Upload a chest X-ray image for AI-powered analysis</p>
              </div>
            </div>
          </motion.div>

          {!prediction ? (
            /*  Upload panel  */
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
              className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800">
              {!selectedImage ? (
                <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300 ${isDragActive ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' : 'border-gray-300 dark:border-gray-700 hover:border-primary-500 dark:hover:border-primary-500'}`}>
                  <input {...getInputProps()} />
                  <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                    {isDragActive ? 'Drop the image here' : 'Upload Chest X-ray'}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">Drag & drop or click to browse</p>
                  <p className="text-xs text-gray-500">PNG  JPG  JPEG</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl overflow-hidden border-2 border-gray-200 dark:border-gray-800">
                    <div className="relative w-full h-64">
                      {imagePreview && <Image src={imagePreview} alt="Selected chest X-ray" fill className="object-contain" />}
                    </div>
                    <button onClick={() => { setSelectedImage(null); setImagePreview(null); }}
                      className="absolute top-2 right-2 p-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors">
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                    <p className="text-sm text-gray-700 dark:text-gray-300"><span className="font-semibold">File:</span> {selectedImage.name}</p>
                    <p className="text-sm text-gray-700 dark:text-gray-300"><span className="font-semibold">Size:</span> {(selectedImage.size / 1024).toFixed(2)} KB</p>
                  </div>
                  <button onClick={handlePredict} disabled={loading}
                    className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-primary-500 to-primary-600 text-white font-semibold hover:from-primary-600 hover:to-primary-700 focus:ring-4 focus:ring-primary-500/50 transition-all duration-300 shadow-lg shadow-primary-500/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2">
                    {loading ? (<><div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white" /> Analysing Image</>) : (<><Wind className="w-5 h-5" /> Detect Pneumonia</>)}
                  </button>
                </div>
              )}
            </motion.div>

          ) : (
            /*  Results panel  */
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="space-y-6">

              {/* Image comparison */}
              {imagePreview && (
                <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-base font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                      <Eye className="w-5 h-5 text-indigo-500" /> Visual Analysis
                    </h3>
                    {gradcamImage && (
                      <div className="flex rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden text-xs font-medium">
                        <button onClick={() => setShowGradcam(false)} className={`px-3 py-1.5 transition-colors ${!showGradcam ? 'bg-primary-500 text-white' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'}`}>Original</button>
                        <button onClick={() => setShowGradcam(true)}  className={`px-3 py-1.5 transition-colors ${showGradcam  ? 'bg-primary-500 text-white' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'}`}>Grad-CAM</button>
                      </div>
                    )}
                  </div>

                  <div className={`grid gap-4 ${gradcamImage ? 'grid-cols-2' : 'grid-cols-1'}`}>
                    {/* Original */}
                    <div className="space-y-2">
                      <p className="text-xs text-center font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Original X-ray</p>
                      <div className="relative w-full h-52 rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                        <Image src={imagePreview} alt="Original chest X-ray" fill className="object-contain" />
                      </div>
                    </div>
                    {/* Grad-CAM */}
                    {gradcamImage && (
                      <div className="space-y-2">
                        <p className="text-xs text-center font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Grad-CAM Heatmap</p>
                        <div className="relative w-full h-52 rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={gradcamImage} alt="Grad-CAM overlay" className="w-full h-full object-contain" />
                        </div>
                        <p className="text-xs text-center text-gray-400 dark:text-gray-500">Red regions  highest model attention (affected lung areas)</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Result card */}
              <div className={`p-8 rounded-2xl border-2 ${isPneumonia ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'}`}>
                <div className="flex items-center gap-4 mb-6">
                  <AlertCircle className={`w-12 h-12 ${isPneumonia ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`} />
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Detection Result</h2>
                    <p className={`text-xl font-bold ${isPneumonia ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>{prediction.prediction}</p>
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Confidence</div>
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">{prediction.confidence?.toFixed(1)}%</div>
                  </div>
                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Normal Probability</div>
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">{prediction.probability?.normal?.toFixed(1)}%</div>
                  </div>
                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Pneumonia Probability</div>
                    <div className="text-2xl font-bold text-red-600 dark:text-red-400">{prediction.probability?.pneumonia?.toFixed(1)}%</div>
                  </div>
                </div>
              </div>

              {/* Model metrics */}
              <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
                <div className="flex items-center gap-3 mb-4">
                  <TrendingUp className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                  <h3 className="text-base font-semibold text-gray-900 dark:text-white">
                    Model Performance  {prediction.architecture ?? 'EfficientNetB0'}
                  </h3>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {[
                    { label: 'Accuracy',  val: prediction.accuracy,  color: 'text-gray-900 dark:text-white' },
                    { label: 'Precision', val: prediction.precision, color: 'text-blue-600 dark:text-blue-400' },
                    { label: 'Recall',    val: prediction.recall,    color: 'text-purple-600 dark:text-purple-400' },
                    { label: 'AUC',       val: prediction.auc,       color: 'text-indigo-600 dark:text-indigo-400' },
                  ].map(({ label, val, color }) => (
                    <div key={label} className="rounded-xl p-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-center">
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">{label}</div>
                      <div className={`text-xl font-bold ${color}`}>{val?.toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
                {prediction.threshold !== undefined && (
                  <p className="mt-3 text-xs text-gray-500 dark:text-gray-400 text-center">
                    Decision threshold: <span className="font-semibold text-gray-700 dark:text-gray-300">{prediction.threshold?.toFixed(4)}</span> (Youden&apos;s J statistic on ROC curve)
                  </p>
                )}
              </div>

              <button onClick={handleReset}
                className="w-full py-4 px-6 rounded-xl bg-white dark:bg-gray-900 border-2 border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-300 font-semibold flex items-center justify-center gap-2">
                <ArrowLeft className="w-5 h-5" /> New Detection
              </button>
            </motion.div>
          )}
        </div>
      </main>
    </div>
  );
}
