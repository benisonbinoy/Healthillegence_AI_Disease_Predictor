'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Heart, TrendingUp, AlertCircle } from 'lucide-react';
import { useAuthStore } from '@/lib/store/authStore';
import Navigation from '@/components/ui/Navigation';
import { motion } from 'framer-motion';

export default function KidneyPredictionPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);

  const [formData, setFormData] = useState({
    age: '',
    bp: '',
    sg: '',
    al: '',
    su: '',
    rbc: 'normal',
    pc: 'normal',
    pcc: 'notpresent',
    ba: 'notpresent',
    bgr: '',
    bu: '',
    sc: '',
    sod: '',
    pot: '',
    hemo: '',
    pcv: '',
    wc: '',
    rc: '',
    htn: 'no',
    dm: 'no',
    cad: 'no',
    appet: 'good',
    pe: 'no',
    ane: 'no',
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
      const response = await fetch('/api/predict/kidney', {
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
      age: '',
      bp: '',
      sg: '',
      al: '',
      su: '',
      rbc: 'normal',
      pc: 'normal',
      pcc: 'notpresent',
      ba: 'notpresent',
      bgr: '',
      bu: '',
      sc: '',
      sod: '',
      pot: '',
      hemo: '',
      pcv: '',
      wc: '',
      rc: '',
      htn: 'no',
      dm: 'no',
      cad: 'no',
      appet: 'good',
      pe: 'no',
      ane: 'no',
    });
  };

  const fields = [
    // Numerical fields
    { name: 'age', label: 'Age', placeholder: 'e.g., 48', type: 'number', step: '1' },
    { name: 'bp', label: 'Blood Pressure (mm/Hg)', placeholder: 'e.g., 80', type: 'number', step: '1' },
    { name: 'sg', label: 'Specific Gravity', placeholder: 'e.g., 1.020', type: 'number', step: '0.001' },
    { name: 'al', label: 'Albumin (0-5)', placeholder: 'e.g., 0', type: 'number', step: '1' },
    { name: 'su', label: 'Sugar (0-5)', placeholder: 'e.g., 0', type: 'number', step: '1' },
    
    // Categorical fields
    { 
      name: 'rbc', 
      label: 'Red Blood Cells', 
      type: 'select',
      options: [
        { value: 'normal', label: 'Normal' },
        { value: 'abnormal', label: 'Abnormal' }
      ]
    },
    { 
      name: 'pc', 
      label: 'Pus Cell', 
      type: 'select',
      options: [
        { value: 'normal', label: 'Normal' },
        { value: 'abnormal', label: 'Abnormal' }
      ]
    },
    { 
      name: 'pcc', 
      label: 'Pus Cell Clumps', 
      type: 'select',
      options: [
        { value: 'notpresent', label: 'Not Present' },
        { value: 'present', label: 'Present' }
      ]
    },
    { 
      name: 'ba', 
      label: 'Bacteria', 
      type: 'select',
      options: [
        { value: 'notpresent', label: 'Not Present' },
        { value: 'present', label: 'Present' }
      ]
    },
    
    { name: 'bgr', label: 'Blood Glucose Random (mgs/dl)', placeholder: 'e.g., 121', type: 'number', step: '1' },
    { name: 'bu', label: 'Blood Urea (mgs/dl)', placeholder: 'e.g., 36', type: 'number', step: '1' },
    { name: 'sc', label: 'Serum Creatinine (mgs/dl)', placeholder: 'e.g., 1.2', type: 'number', step: '0.1' },
    { name: 'sod', label: 'Sodium (mEq/L)', placeholder: 'e.g., 140', type: 'number', step: '1' },
    { name: 'pot', label: 'Potassium (mEq/L)', placeholder: 'e.g., 4.5', type: 'number', step: '0.1' },
    { name: 'hemo', label: 'Hemoglobin (gms)', placeholder: 'e.g., 15.4', type: 'number', step: '0.1' },
    { name: 'pcv', label: 'Packed Cell Volume', placeholder: 'e.g., 44', type: 'number', step: '1' },
    { name: 'wc', label: 'White Blood Cell Count (cells/cumm)', placeholder: 'e.g., 7800', type: 'number', step: '100' },
    { name: 'rc', label: 'Red Blood Cell Count (millions/cmm)', placeholder: 'e.g., 5.2', type: 'number', step: '0.1' },
    
    { 
      name: 'htn', 
      label: 'Hypertension', 
      type: 'select',
      options: [
        { value: 'no', label: 'No' },
        { value: 'yes', label: 'Yes' }
      ]
    },
    { 
      name: 'dm', 
      label: 'Diabetes Mellitus', 
      type: 'select',
      options: [
        { value: 'no', label: 'No' },
        { value: 'yes', label: 'Yes' }
      ]
    },
    { 
      name: 'cad', 
      label: 'Coronary Artery Disease', 
      type: 'select',
      options: [
        { value: 'no', label: 'No' },
        { value: 'yes', label: 'Yes' }
      ]
    },
    { 
      name: 'appet', 
      label: 'Appetite', 
      type: 'select',
      options: [
        { value: 'good', label: 'Good' },
        { value: 'poor', label: 'Poor' }
      ]
    },
    { 
      name: 'pe', 
      label: 'Pedal Edema', 
      type: 'select',
      options: [
        { value: 'no', label: 'No' },
        { value: 'yes', label: 'Yes' }
      ]
    },
    { 
      name: 'ane', 
      label: 'Anemia', 
      type: 'select',
      options: [
        { value: 'no', label: 'No' },
        { value: 'yes', label: 'Yes' }
      ]
    },
  ];

  // Check if required numerical fields are filled
  const requiredFields = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'];
  const isFormValid = requiredFields.every((field) => formData[field as keyof typeof formData] !== '');

  if (!isAuthenticated) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-purple-50 to-gray-100 dark:from-gray-950 dark:via-purple-950/20 dark:to-gray-900">
      <Navigation />

      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto">
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
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Kidney Disease Prediction
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                  Enter complete patient kidney function test results (98.75% accuracy)
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
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
                {fields.map((field) => (
                  <div key={field.name} className={field.type === 'select' ? 'md:col-span-1' : ''}>
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
                        {field.options?.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        name={field.name}
                        value={formData[field.name as keyof typeof formData]}
                        onChange={handleInputChange}
                        placeholder={field.placeholder}
                        step={field.step || 'any'}
                        className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                      />
                    )}
                  </div>
                ))}
              </div>

              <button
                onClick={handlePredict}
                disabled={loading || !isFormValid}
                className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-purple-500 to-pink-600 text-white font-semibold hover:from-purple-600 hover:to-pink-700 focus:ring-4 focus:ring-purple-500/50 transition-all duration-300 shadow-lg shadow-purple-500/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Heart className="w-5 h-5" />
                    Predict Kidney Disease Risk
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
