import React, { useEffect, useState } from 'react';
import { Shield, Activity, Cpu } from 'lucide-react';
import { checkHealth } from '../services/api';

export default function Navbar() {
  const [engineStatus, setEngineStatus] = useState('checking'); // 'online' | 'offline' | 'checking'

  useEffect(() => {
    let isMounted = true;
    checkHealth()
      .then((res) => {
        if (isMounted) {
          setEngineStatus(res.status === 'ok' ? 'online' : 'offline');
        }
      })
      .catch(() => {
        if (isMounted) {
          setEngineStatus('offline');
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 p-0.5 flex items-center justify-center shadow-lg shadow-emerald-950/40">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Shield className="w-5 h-5 text-emerald-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white font-sans">
                Veritas<span className="text-emerald-400">AI</span>
              </span>
              <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                V1.0
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Evidence-backed claim verification
            </p>
          </div>
        </div>

        {/* Engine status indicator */}
        <div className="flex items-center space-x-2 text-xs">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800">
            <span className="relative flex h-2 w-2">
              {engineStatus === 'online' && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  engineStatus === 'online'
                    ? 'bg-emerald-500'
                    : engineStatus === 'offline'
                    ? 'bg-rose-500'
                    : 'bg-amber-400'
                }`}
              ></span>
            </span>
            <span className="font-medium text-slate-300 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              {engineStatus === 'online'
                ? 'AI Verification Engine Active'
                : engineStatus === 'offline'
                ? 'Backend Disconnected'
                : 'Connecting to Engine...'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
