import React, { useState, useEffect } from 'react';
import { Search, Globe, FileText, Brain, Sparkles, Loader2 } from 'lucide-react';

const STAGES = [
  { icon: Search, label: 'Analyzing input & extracting claims...', detail: 'Scanning content for verifiable factual assertions' },
  { icon: Globe, label: 'Searching the web with Google Search grounding...', detail: 'Retrieving official records, wire reports, and corroborating sources' },
  { icon: FileText, label: 'Reviewing and filtering sources...', detail: 'Comparing multiple independent reports and publication dates' },
  { icon: Brain, label: 'Gemini 3.7 Flash evaluating evidence...', detail: 'Checking for context distortion, temporal shifts, and contradictions' },
  { icon: Sparkles, label: 'Synthesizing evidence-based verdict...', detail: 'Formatting structured verification response' },
];

export default function GeminiLoadingState() {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentStage((prev) => (prev < STAGES.length - 1 ? prev + 1 : prev));
    }, 2800);
    return () => clearInterval(timer);
  }, []);

  const active = STAGES[currentStage];
  const ActiveIcon = active.icon;

  return (
    <div className="w-full max-w-3xl mx-auto p-6 sm:p-8 rounded-2xl bg-slate-900/90 border border-cyan-500/30 shadow-2xl shadow-cyan-950/20 backdrop-blur-md animate-fade-in text-center space-y-6">
      {/* Glow / Spinner */}
      <div className="relative inline-flex items-center justify-center">
        <div className="absolute inset-0 rounded-full bg-cyan-500/20 blur-xl animate-pulse" />
        <div className="relative p-4 rounded-2xl bg-slate-950/80 border border-cyan-500/40 text-cyan-400">
          <ActiveIcon className="w-8 h-8 animate-bounce transition-all duration-300" />
        </div>
      </div>

      {/* Stage Titles */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-center gap-2 text-xs font-semibold text-cyan-400 uppercase tracking-wider font-mono">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          <span>Gemini 3.7 Flash Investigation</span>
        </div>
        <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-sans">
          {active.label}
        </h3>
        <p className="text-sm text-slate-400 max-w-md mx-auto">
          {active.detail}
        </p>
      </div>

      {/* Stage indicators */}
      <div className="flex justify-center items-center gap-2 pt-2">
        {STAGES.map((s, idx) => (
          <div
            key={idx}
            className={`h-1.5 rounded-full transition-all duration-500 ${
              idx === currentStage
                ? 'w-8 bg-cyan-400 shadow-sm shadow-cyan-400'
                : idx < currentStage
                ? 'w-4 bg-cyan-600/60'
                : 'w-2 bg-slate-800'
            }`}
          />
        ))}
      </div>
    </div>
  );
}
