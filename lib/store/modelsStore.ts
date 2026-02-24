import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface ModelInfo {
  accuracy: number;
  last_trained: string | null;
  features?: string[];
}

interface ModelsState {
  diabetes: ModelInfo;
  kidney: ModelInfo;
  liver: ModelInfo;
  malaria: ModelInfo;
  pneumonia: ModelInfo;
}

interface ModelsStore {
  models: ModelsState;
  loading: boolean;
  error: string | null;
  lastFetched: number | null;
  fetchModelInfo: () => Promise<void>;
  clearCache: () => void;
}

export const useModelsStore = create<ModelsStore>()(
  persist(
    (set) => ({
      models: {
        diabetes: { accuracy: 0, last_trained: null },
        kidney: { accuracy: 0, last_trained: null },
        liver: { accuracy: 0, last_trained: null },
        malaria: { accuracy: 0, last_trained: null },
        pneumonia: { accuracy: 0, last_trained: null },
      },
      loading: false,
      error: null,
      lastFetched: null,
      fetchModelInfo: async () => {
        set({ loading: true, error: null });
        try {
          // Add cache-busting query parameter
          const timestamp = new Date().getTime();
          const response = await fetch(`/api/model-info?t=${timestamp}`);
          const result = await response.json();
          
          if (result.success) {
            set({ models: result.data, loading: false, lastFetched: timestamp });
          } else {
            set({ error: result.error, loading: false });
          }
        } catch (error) {
          set({ error: 'Failed to fetch model information', loading: false });
        }
      },
      clearCache: () => {
        set({
          models: {
            diabetes: { accuracy: 0, last_trained: null },
            kidney: { accuracy: 0, last_trained: null },
            liver: { accuracy: 0, last_trained: null },
            malaria: { accuracy: 0, last_trained: null },
            pneumonia: { accuracy: 0, last_trained: null },
          },
          lastFetched: null,
        });
      },
    }),
    {
      name: 'models-storage',
      version: 2, // Increment version to invalidate old cache
    }
  )
);
