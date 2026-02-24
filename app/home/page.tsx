'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Activity, Brain, Droplet, Heart, Bug, Wind, TrendingUp } from 'lucide-react';
import { useAuthStore } from '@/lib/store/authStore';
import { useModelsStore } from '@/lib/store/modelsStore';
import Navigation from '@/components/ui/Navigation';
import { motion } from 'framer-motion';

export default function HomePage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();
  const { models, fetchModelInfo } = useModelsStore();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/signin');
      return;
    }
    fetchModelInfo();
  }, [isAuthenticated, router, fetchModelInfo]);

  const modelCards = [
    {
      name: 'Diabetes',
      icon: Droplet,
      description: 'Predict diabetes risk based on health metrics',
      accuracy: models.diabetes.accuracy,
      color: 'from-blue-500 to-cyan-500',
      route: '/prediction/diabetes',
    },
    {
      name: 'Kidney Disease',
      icon: Heart,
      description: 'Analyze kidney function and disease probability',
      accuracy: models.kidney.accuracy,
      color: 'from-purple-500 to-pink-500',
      route: '/prediction/kidney',
    },
    {
      name: 'Liver Disease',
      icon: Activity,
      description: 'Assess liver health and potential conditions',
      accuracy: models.liver.accuracy,
      color: 'from-orange-500 to-red-500',
      route: '/prediction/liver',
    },
    {
      name: 'Malaria',
      icon: Bug,
      description: 'Detect malaria from blood cell images',
      accuracy: models.malaria.accuracy,
      color: 'from-green-500 to-emerald-500',
      route: '/prediction/malaria',
    },
    {
      name: 'Pneumonia',
      icon: Wind,
      description: 'Identify pneumonia from chest X-ray images',
      accuracy: models.pneumonia.accuracy,
      color: 'from-indigo-500 to-blue-500',
      route: '/prediction/pneumonia',
    },
  ];

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-red-50 to-gray-100 dark:from-gray-950 dark:via-red-950/20 dark:to-gray-900">
      <Navigation />

      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 mb-4">
              <Brain className="w-5 h-5" />
              <span className="font-semibold">AI-Powered Health Analysis</span>
            </div>
            
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
              Welcome to{' '}
              <span className="bg-gradient-to-r from-primary-500 to-primary-600 bg-clip-text text-transparent">
                Healthillegence
              </span>
            </h1>
            
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mb-8">
              Advanced machine learning models trained on extensive medical datasets to provide
              accurate health predictions and assist in early disease detection.
            </p>
          </motion.div>

          {/* Model Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {modelCards.map((model, index) => {
              const Icon = model.icon;
              return (
                <motion.div
                  key={model.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="bg-white dark:bg-gray-900 rounded-2xl p-6 border border-gray-200 dark:border-gray-800 hover:shadow-xl hover:shadow-primary-500/10 transition-all duration-300 group cursor-pointer"
                  onClick={() => router.push(model.route)}
                >
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${model.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                    {model.name}
                  </h3>
                  
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {model.description}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      Model Accuracy
                    </span>
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-green-500" />
                      <span className="text-lg font-bold text-gray-900 dark:text-white">
                        {model.accuracy > 0 ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* Features Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800"
          >
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              Current Features
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center flex-shrink-0">
                  <Droplet className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                    Numerical Predictions
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Diabetes, kidney, and liver disease predictions based on health metrics
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center flex-shrink-0">
                  <Brain className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                    Image Analysis
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Malaria and pneumonia detection using advanced CNN models
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center flex-shrink-0">
                  <TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                    High Accuracy
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Models trained on extensive datasets with regular updates
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center flex-shrink-0">
                  <Activity className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                    Real-time Analysis
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Instant predictions with confidence scores and probabilities
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
