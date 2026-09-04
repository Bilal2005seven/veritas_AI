import React from 'react';
import { Search, ShieldAlert, Cpu, CheckCircle } from 'lucide-react';

export default function EmptyState() {
  return (
    <div className="border border-dashed border-slate-800 rounded-2xl p-8 sm:p-14 text-center bg-slate-900/20 backdrop-blur-sm">
      <div className="max-w-md mx-auto">
        <div className="w-14 h-14 mx-auto rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-4 text-emerald-400">
          <Search className="w-6 h-6 text-emerald-400" />
        </div>

        <h3 className="text-lg sm:text-xl font-bold text-white mb-2 tracking-tight">
          Ready to verify
        </h3>
        <p className="text-xs sm:text-sm text-slate-400 leading-relaxed mb-6">
          Enter a factual claim above or pick a demo prompt. VeritasAI will retrieve real-time web evidence, analyze stance using Natural Language Inference, and evaluate credibility scores.
        </p>

        {/* Feature badges */}
        <div className="flex flex-wrap justify-center gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800/80">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span>MiniLM Embeddings</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800/80">
            <ShieldAlert className="w-3.5 h-3.5 text-teal-400" />
            <span>NLI Cross-Encoder</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800/80">
            <CheckCircle className="w-3.5 h-3.5 text-indigo-400" />
            <span>Domain Authority Scoring</span>
          </div>
        </div>
      </div>
    </div>
  );
}
