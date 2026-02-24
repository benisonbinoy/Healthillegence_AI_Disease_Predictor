import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  username: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => boolean;
  signup: (username: string, password: string) => boolean;
  logout: () => void;
}

// Simple demo authentication - replace with real auth in production
export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      login: (username: string, password: string) => {
        // Demo login - accept any non-empty credentials
        if (username && password) {
          set({ user: { username }, isAuthenticated: true });
          return true;
        }
        return false;
      },
      signup: (username: string, password: string) => {
        // Demo signup - accept any non-empty credentials
        if (username && password) {
          set({ user: { username }, isAuthenticated: true });
          return true;
        }
        return false;
      },
      logout: () => {
        set({ user: null, isAuthenticated: false });
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);
