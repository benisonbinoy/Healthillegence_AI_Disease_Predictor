'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { User, Mail, Award, Github, Linkedin, Globe } from 'lucide-react';
import { useAuthStore } from '@/lib/store/authStore';
import Navigation from '@/components/ui/Navigation';
import { motion } from 'framer-motion';

export default function AboutPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/signin');
    }
  }, [isAuthenticated, router]);

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
            className="text-center mb-12"
          >
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
              About{' '}
              <span className="bg-gradient-to-r from-primary-500 to-primary-600 bg-clip-text text-transparent">
                Healthiligence
              </span>
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              An advanced AI-powered health prediction system for early disease detection
            </p>
          </motion.div>

          {/* Project Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800 mb-8"
          >
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
              <Globe className="w-7 h-7 text-primary-600 dark:text-primary-400" />
              Project Overview
            </h2>
            <div className="space-y-4 text-gray-700 dark:text-gray-300">
              <p>
                Healthiligence is a comprehensive health analysis platform that leverages machine
                learning and deep learning technologies to provide accurate predictions for various
                diseases. The system is designed to assist healthcare professionals and individuals
                in early disease detection and risk assessment.
              </p>
              <p>
                The platform integrates multiple predictive models trained on extensive medical
                datasets from Kaggle, covering both numerical health metrics and medical imaging
                analysis.
              </p>
            </div>
          </motion.div>

          {/* Features */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800 mb-8"
          >
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
              <Award className="w-7 h-7 text-primary-600 dark:text-primary-400" />
              Key Features
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  • Diabetes Prediction
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 ml-4">
                  Based on Pima Indians Diabetes Database with 8 key health metrics
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  • Kidney Disease Detection
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 ml-4">
                  Comprehensive analysis using 24 clinical parameters
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  • Liver Disease Assessment
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 ml-4">
                  Evaluation based on liver function tests and patient demographics
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  • Malaria Detection
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 ml-4">
                  CNN-based analysis of blood cell images for parasite detection
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  • Pneumonia Detection
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 ml-4">
                  Deep learning model for chest X-ray image classification
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  • Real-time Analysis
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 ml-4">
                  Instant predictions with confidence scores and probabilities
                </p>
              </div>
            </div>
          </motion.div>

          {/* Project Maker */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gradient-to-r from-primary-500 to-primary-600 rounded-2xl p-8 text-white"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <User className="w-7 h-7" />
              Project Maker
            </h2>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <div className="flex items-center gap-6 mb-6">
                <div className="w-20 h-20 rounded-full bg-white/20 flex items-center justify-center">
                  <User className="w-10 h-10" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold">Benison Binoy</h3>
                  <p className="text-white/90">Registration Number: 2262044</p>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-3 text-white/90">
                  <Award className="w-5 h-5" />
                  <span>Final Year Project - 2026</span>
                </div>
                <div className="flex items-center gap-3 text-white/90">
                  <Mail className="w-5 h-5" />
                  <span>Computer Science & Engineering (AI &ML)</span>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-white/20">
                <p className="text-sm text-white/80">
                  This project demonstrates the application of machine learning and deep learning
                  in healthcare, specifically focusing on disease prediction and medical image
                  analysis for early detection and risk assessment.
                </p>
              </div>

              {/* Social Links Placeholder */}
              <div className="mt-6 flex gap-4">
                <button className="p-3 rounded-lg bg-white/10 hover:bg-white/20 transition-all">
                  <Github className="w-5 h-5" />
                </button>
                <button className="p-3 rounded-lg bg-white/10 hover:bg-white/20 transition-all">
                  <Linkedin className="w-5 h-5" />
                </button>
                <button className="p-3 rounded-lg bg-white/10 hover:bg-white/20 transition-all">
                  <Mail className="w-5 h-5" />
                </button>
              </div>
            </div>
          </motion.div>

          {/* Technologies Used */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-8 bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800"
          >
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              Technologies Used
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {[
                'Next.js',
                'TensorFlow',
                'Python',
                'React',
                'scikit-learn',
                'Keras',
                'Flask',
                'Tailwind CSS',
                'TypeScript',
              ].map((tech) => (
                <div
                  key={tech}
                  className="px-4 py-3 rounded-lg bg-gray-100 dark:bg-gray-800 text-center font-medium text-gray-700 dark:text-gray-300"
                >
                  {tech}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-8 p-6 rounded-xl bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800"
          >
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <strong>Disclaimer:</strong> This system is designed for educational and research
              purposes. The predictions should not be used as a substitute for professional medical
              advice, diagnosis, or treatment. Always consult with qualified healthcare professionals
              for medical concerns.
            </p>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
