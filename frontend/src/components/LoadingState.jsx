import React, { useEffect, useState } from 'react';
import { Loader2, Search, Brain, GitCompare, Scale } from 'lucide-react';

const STAGES = [
  { icon: Search, label: 'Searching live web for evidence...' },
  { icon: Brain, label: 'Evaluating semantic relevance with MiniLM...' },
  { icon: GitCompare, label: 'Cross-encoding NLI stance relationships...' },
  { icon: Scale, label: 'Calculating domain & evidence credibility scores...' },
];

export default function LoadingState() {
  const [currentStageIdx, setCurrentStageIdx] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStageIdx((prev) => (prev + 1) % STAGES.length);
    }, 2800);
    return () => clearInterval(interval);
  }, []);

  const ActiveIcon = STAGES[currentStageIdx].icon;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 sm:p-12 text-center backdrop-blur-sm shadow-xl animate-fade-in">
      <div className="max-w-md mx-auto flex flex-col items-center">
        {/* Animated radar/spinner ring */}
        <div className="relative mb-6">
          <div className="w-16 h-16 rounded-full border-2 border-emerald-500/20 border-t-emerald-500 animate-spin flex items-center justify-center">
            <ActiveIcon className="w-6 h-6 text-emerald-400 transition-all duration-300" />
          </div>
          <div className="absolute inset-0 rounded-full bg-emerald-500/10 blur-md -z-10 animate-pulse" />
        </div>

        <h2 className="text-xl font-bold text-white mb-2 tracking-tight">
          Analyzing Claim...
        </h2>
        <p className="text-sm text-emerald-400 font-mono flex items-center gap-2 h-6 mb-6">
          <Loader2 className="w-3.5 h-3.5 animate-spin inline" />
          <span>{STAGES[currentStageIdx].label}</span>
        </p>

        {/* Pipeline indicators */}
        <div className="w-full grid grid-cols-2 sm:grid-cols-4 gap-2 pt-4 border-t border-slate-800/80 text-[11px] text-slate-500">
          {STAGES.map((s, idx) => {
            const isPassed = idx <= currentStageIdx;
            const isCurrent = idx === currentStageIdx;
            return (
              <div
                key={idx}
                className={`p-2 rounded-lg border text-center transition-colors ${
                  isCurrent
                    ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300'
                    : isPassed
                    ? 'border-slate-800 bg-slate-900/40 text-slate-400'
                    : 'border-slate-900 bg-slate-950/40 text-slate-600'
                }`}
              >
                {idx + 1}. {s.label.split(' ')[0]}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
