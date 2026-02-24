'use client';

import Link from 'next/link';
import Image from 'next/image';
import { usePathname, useRouter } from 'next/navigation';
import { Home, Activity, Info, Settings, LogOut } from 'lucide-react';
import { useAuthStore } from '@/lib/store/authStore';
import ThemeToggle from './ThemeToggle';
import logo from '@/images/logo.png';
import logo2 from '@/images/logo2.png';

export default function Navigation() {
  const pathname = usePathname();
  const router = useRouter();
  const { logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    router.push('/signin');
  };

  const navItems = [
    { name: 'Home', href: '/home', icon: Home },
    { name: 'About', href: '/about', icon: Info },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-gray-950/80 backdrop-blur-lg border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/home" className="flex items-center gap-3 group">
            <div className="relative w-10 h-10 rounded-lg overflow-hidden ring-2 ring-primary-500/20 group-hover:ring-primary-500/50 transition-all">
              <Image
                src={logo}
                alt="Healthiligence Logo"
                fill
                className="object-contain"
              />
            </div>
            <div className="relative w-32 h-8 hidden sm:block">
              <Image
                src={logo2}
                alt="Healthiligence"
                fill
                className="object-contain"
              />
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                    isActive
                      ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-lg shadow-primary-500/30'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden md:inline">{item.name}</span>
                </Link>
              );
            })}

            {/* Theme Toggle */}
            <ThemeToggle />

            {/* Logout Button */}
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-gray-700 dark:text-gray-300 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 dark:hover:text-red-400 transition-all duration-300"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden md:inline">Sign Out</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
