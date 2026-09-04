import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default function ErrorState({ message, onRetry }) {
  return (
    <div className="rounded-2xl border border-rose-500/30 bg-rose-950/20 p-6 sm:p-8 backdrop-blur-sm shadow-xl text-center">
      <div className="max-w-md mx-auto flex flex-col items-center">
        <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-4">
          <AlertCircle className="w-6 h-6 text-rose-400" />
        </div>

        <h3 className="text-lg font-bold text-white mb-1.5 tracking-tight">
          Verification Request Failed
        </h3>

        <p className="text-sm text-rose-300/90 leading-relaxed mb-6">
          {message || 'Something went wrong while verifying this claim.'}
        </p>

        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-sm font-semibold text-white transition-colors focus:outline-none focus:ring-1 focus:ring-rose-400"
          >
            <RefreshCw className="w-4 h-4 text-rose-400" />
            <span>Try Again</span>
          </button>
        )}
      </div>
    </div>
  );
}
