'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Activity, TrendingUp, AlertCircle } from 'lucide-react';
import { useAuthStore } from '@/lib/store/authStore';
import Navigation from '@/components/ui/Navigation';
import { motion } from 'framer-motion';

export default function LiverPredictionPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);

  const [formData, setFormData] = useState({
    Age: '',
    Gender: '1',
    Total_Bilirubin: '',
    Direct_Bilirubin: '',
    Alkaline_Phosphotase: '',
    Alamine_Aminotransferase: '',
    Aspartate_Aminotransferase: '',
    Total_Protiens: '',
    Albumin: '',
    Albumin_and_Globulin_Ratio: '',
  });

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/signin');
    }
  }, [isAuthenticated, router]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/predict/liver', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setFormData({
      Age: '',
      Gender: '1',
      Total_Bilirubin: '',
      Direct_Bilirubin: '',
      Alkaline_Phosphotase: '',
      Alamine_Aminotransferase: '',
      Aspartate_Aminotransferase: '',
      Total_Protiens: '',
      Albumin: '',
      Albumin_and_Globulin_Ratio: '',
    });
  };

  const fields = [
    { name: 'Age', label: 'Age (years)', placeholder: 'e.g., 45', type: 'number' },
    { name: 'Gender', label: 'Gender', type: 'select' },
    { name: 'Total_Bilirubin', label: 'Total Bilirubin (mg/dL)', placeholder: 'e.g., 0.7', type: 'number' },
    { name: 'Direct_Bilirubin', label: 'Direct Bilirubin (mg/dL)', placeholder: 'e.g., 0.2', type: 'number' },
    { name: 'Alkaline_Phosphotase', label: 'Alkaline Phosphotase (IU/L)', placeholder: 'e.g., 187', type: 'number' },
    { name: 'Alamine_Aminotransferase', label: 'Alamine Aminotransferase (IU/L)', placeholder: 'e.g., 16', type: 'number' },
    { name: 'Aspartate_Aminotransferase', label: 'Aspartate Aminotransferase (IU/L)', placeholder: 'e.g., 18', type: 'number' },
    { name: 'Total_Protiens', label: 'Total Proteins (g/dL)', placeholder: 'e.g., 6.8', type: 'number' },
    { name: 'Albumin', label: 'Albumin (g/dL)', placeholder: 'e.g., 3.3', type: 'number' },
    { name: 'Albumin_and_Globulin_Ratio', label: 'Albumin and Globulin Ratio', placeholder: 'e.g., 0.9', type: 'number' },
  ];

  if (!isAuthenticated) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-red-50 to-gray-100 dark:from-gray-950 dark:via-red-950/20 dark:to-gray-900">
      <Navigation />

      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <button
              onClick={() => router.push('/home')}
              className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 mb-4"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </button>

            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-orange-500 to-red-500 flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Liver Disease Prediction
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                  Enter patient liver function test results
                </p>
              </div>
            </div>
          </motion.div>

          {!prediction ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {fields.map((field) => (
                  <div key={field.name}>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      {field.label}
                    </label>
                    {field.type === 'select' ? (
                      <select
                        name={field.name}
                        value={formData[field.name as keyof typeof formData]}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                      >
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                      </select>
                    ) : (
                      <input
                        type="number"
                        name={field.name}
                        value={formData[field.name as keyof typeof formData]}
                        onChange={handleInputChange}
                        placeholder={field.placeholder}
                        step="any"
                        className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                      />
                    )}
                  </div>
                ))}
              </div>

              <button
                onClick={handlePredict}
                disabled={loading || Object.entries(formData).some(([k, v]) => k !== 'Gender' && !v)}
                className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-primary-500 to-primary-600 text-white font-semibold hover:from-primary-600 hover:to-primary-700 focus:ring-4 focus:ring-primary-500/50 transition-all duration-300 shadow-lg shadow-primary-500/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="w-5 h-5" />
                    Predict Liver Disease Risk
                  </>
                )}
              </button>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-6"
            >
              {/* Result Card */}
              <div className={`p-8 rounded-2xl border-2 ${
                prediction.prediction === 'Positive'
                  ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                  : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              }`}>
                <div className="flex items-center gap-4 mb-4">
                  <AlertCircle className={`w-12 h-12 ${
                    prediction.prediction === 'Positive'
                      ? 'text-red-600 dark:text-red-400'
                      : 'text-green-600 dark:text-green-400'
                  }`} />
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                      Prediction Result
                    </h2>
                    <p className={`text-lg font-semibold ${
                      prediction.prediction === 'Positive'
                        ? 'text-red-600 dark:text-red-400'
                        : 'text-green-600 dark:text-green-400'
                    }`}>
                      {prediction.prediction}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                      Confidence
                    </div>
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">
                      {prediction.confidence.toFixed(1)}%
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                      Negative Probability
                    </div>
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {prediction.probability.negative.toFixed(1)}%
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                      Positive Probability
                    </div>
                    <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                      {prediction.probability.positive.toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Model Accuracy */}
              <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                    <div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Model Accuracy
                      </div>
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {prediction.accuracy.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Back Button */}
              <button
                onClick={handleReset}
                className="w-full py-4 px-6 rounded-xl bg-white dark:bg-gray-900 border-2 border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-300 font-semibold flex items-center justify-center gap-2"
              >
                <ArrowLeft className="w-5 h-5" />
                New Prediction
              </button>
            </motion.div>
          )}
        </div>
      </main>
    </div>
  );
}
