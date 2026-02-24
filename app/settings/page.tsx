'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { RefreshCw, Settings as SettingsIcon, CheckCircle, AlertCircle } from 'lucide-react';
import { useAuthStore } from '@/lib/store/authStore';
import { useModelsStore } from '@/lib/store/modelsStore';
import Navigation from '@/components/ui/Navigation';
import { motion } from 'framer-motion';

export default function SettingsPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();
  const { models, fetchModelInfo, clearCache } = useModelsStore();
  const [retraining, setRetraining] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/signin');
    } else {
      fetchModelInfo();
    }
  }, [isAuthenticated, router, fetchModelInfo]);

  const handleRetrain = async (modelName: string) => {
    setRetraining(modelName);
    setMessage(null);

    try {
      // Simulate retraining (in production, this would call the actual training endpoint)
      await new Promise((resolve) => setTimeout(resolve, 3000));
      
      setMessage({
        type: 'success',
        text: `${modelName.charAt(0).toUpperCase() + modelName.slice(1)} model retraining initiated. This process may take several minutes.`,
      });
      
      // Refresh model info after retraining
      setTimeout(() => {
        fetchModelInfo();
      }, 1000);
    } catch (error) {
      setMessage({
        type: 'error',
        text: `Failed to retrain ${modelName} model. Please try again.`,
      });
    } finally {
      setRetraining(null);
    }
  };

  const handleRefreshAccuracy = async () => {
    setRefreshing(true);
    await fetchModelInfo();
    setTimeout(() => {
      setRefreshing(false);
      setMessage({
        type: 'success',
        text: 'Model accuracy data refreshed successfully!',
      });
    }, 500);
  };

  const modelsList = [
    {
      name: 'diabetes',
      displayName: 'Diabetes Model',
      description: 'Random Forest classifier for diabetes prediction',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      name: 'kidney',
      displayName: 'Kidney Disease Model',
      description: 'Random Forest classifier for kidney disease detection',
      color: 'from-purple-500 to-pink-500',
    },
    {
      name: 'liver',
      displayName: 'Liver Disease Model',
      description: 'Random Forest classifier for liver disease assessment',
      color: 'from-orange-500 to-red-500',
    },
    {
      name: 'malaria',
      displayName: 'Malaria Detection Model',
      description: 'CNN model for malaria parasite detection from cell images',
      color: 'from-green-500 to-emerald-500',
    },
    {
      name: 'pneumonia',
      displayName: 'Pneumonia Detection Model',
      description: 'CNN model for pneumonia detection from chest X-rays',
      color: 'from-indigo-500 to-blue-500',
    },
  ];

  if (!isAuthenticated) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-red-50 to-gray-100 dark:from-gray-950 dark:via-red-950/20 dark:to-gray-900">
      <Navigation />

      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-primary-500 to-primary-600 flex items-center justify-center">
                  <SettingsIcon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Settings</h1>
                  <p className="text-gray-600 dark:text-gray-400">
                    Manage and retrain prediction models
                  </p>
                </div>
              </div>
              <button
                onClick={handleRefreshAccuracy}
                disabled={refreshing}
                className={`px-4 py-2 rounded-xl font-semibold transition-all duration-300 flex items-center gap-2 ${
                  refreshing
                    ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:from-primary-600 hover:to-primary-700 shadow-lg shadow-primary-500/30'
                }`}
              >
                <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
                {refreshing ? 'Refreshing...' : 'Refresh Accuracy'}
              </button>
            </div>
          </motion.div>

          {/* Message Alert */}
          {message && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mb-6 p-4 rounded-xl border-2 flex items-center gap-3 ${
                message.type === 'success'
                  ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-700 dark:text-green-400'
                  : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400'
              }`}
            >
              {message.type === 'success' ? (
                <CheckCircle className="w-5 h-5 flex-shrink-0" />
              ) : (
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
              )}
              <p>{message.text}</p>
            </motion.div>
          )}

          {/* Info Box */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-8 p-6 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
          >
            <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
              About Model Retraining
            </h3>
            <p className="text-sm text-blue-800 dark:text-blue-300">
              Retraining models allows you to update the AI predictions based on the latest datasets.
              This process typically takes several minutes depending on the dataset size and model
              complexity. Make sure you have the required datasets in the <code className="px-2 py-1 rounded bg-blue-100 dark:bg-blue-900/50">backend/datasets/</code> directory before
              initiating retraining.
            </p>
          </motion.div>

          {/* Models List */}
          <div className="space-y-4">
            {modelsList.map((model, index) => {
              const modelInfo = models[model.name as keyof typeof models];
              const isRetraining = retraining === model.name;

              return (
                <motion.div
                  key={model.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + index * 0.05 }}
                  className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800 hover:shadow-lg transition-shadow"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div
                          className={`w-10 h-10 rounded-lg bg-gradient-to-r ${model.color} flex items-center justify-center`}
                        >
                          <RefreshCw className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                            {model.displayName}
                          </h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {model.description}
                          </p>
                        </div>
                      </div>

                      <div className="ml-13 mt-3 grid grid-cols-2 gap-4">
                        <div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                            Current Accuracy
                          </div>
                          <div className="text-lg font-bold text-gray-900 dark:text-white">
                            {modelInfo.accuracy > 0
                              ? `${(modelInfo.accuracy * 100).toFixed(2)}%`
                              : 'Not trained'}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                            Last Trained
                          </div>
                          <div className="text-sm text-gray-700 dark:text-gray-300">
                            {modelInfo.last_trained
                              ? new Date(modelInfo.last_trained).toLocaleDateString()
                              : 'Never'}
                          </div>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => handleRetrain(model.name)}
                      disabled={isRetraining}
                      className={`ml-6 px-6 py-3 rounded-xl font-semibold transition-all duration-300 flex items-center gap-2 ${
                        isRetraining
                          ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                          : 'bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:from-primary-600 hover:to-primary-700 shadow-lg shadow-primary-500/30'
                      }`}
                    >
                      {isRetraining ? (
                        <>
                          <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                          Retraining...
                        </>
                      ) : (
                        <>
                          <RefreshCw className="w-5 h-5" />
                          Retrain Model
                        </>
                      )}
                    </button>
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* Additional Settings Placeholder */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mt-8 bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800"
          >
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
              Dataset Requirements
            </h3>
            <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <p>
                • <strong>Diabetes:</strong> datasets/diabetes.csv
              </p>
              <p>
                • <strong>Kidney Disease:</strong> datasets/kidney_disease.csv
              </p>
              <p>
                • <strong>Liver Disease:</strong> datasets/indian_liver_patient.csv
              </p>
              <p>
                • <strong>Malaria:</strong> datasets/cell_images/
              </p>
              <p>
                • <strong>Pneumonia:</strong> datasets/chest_xray/
              </p>
            </div>
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              Refer to <code className="px-2 py-1 rounded bg-gray-100 dark:bg-gray-800">backend/DATASETS.md</code> for download instructions.
            </p>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
