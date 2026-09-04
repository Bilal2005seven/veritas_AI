import React from 'react';
import { Sparkles, Bot } from 'lucide-react';

export default function ExplanationCard({ explanation }) {
  if (!explanation) return null;

  return (
    <div className="rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-slate-900/90 via-slate-900/60 to-emerald-950/10 p-6 sm:p-7 backdrop-blur-sm shadow-lg">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-emerald-400" />
          </div>
          <h3 className="text-base sm:text-lg font-bold text-white tracking-tight">
            Why this verdict?
          </h3>
        </div>
        <span className="text-[11px] font-mono text-emerald-400/90 bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded-full flex items-center gap-1">
          <Bot className="w-3 h-3" />
          Synthesized Analysis
        </span>
      </div>

      <div className="text-sm sm:text-base text-slate-300 leading-relaxed font-sans whitespace-pre-line pl-1 border-l-2 border-emerald-500/30 ml-2 py-0.5">
        {explanation}
      </div>
    </div>
  );
}
