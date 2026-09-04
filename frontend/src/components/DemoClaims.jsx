import React from 'react';
import { Sparkles } from 'lucide-react';

const DEMO_CLAIMS = [
  {
    label: 'Scientific Fact',
    text: 'The Earth revolves around the Sun.',
    badge: 'Standard Truth',
  },
  {
    label: 'Local Incident',
    text: 'Kal Damoh ke post office ke paas do cars ka accident hua tha.',
    badge: 'Hyperlocal Claim',
  },
  {
    label: 'Physical Anomaly',
    text: 'The Sun is made of solid iron.',
    badge: 'Contradicted Claim',
  },
];

export default function DemoClaims({ onSelectClaim, disabled }) {
  return (
    <div className="mt-4 pt-4 border-t border-slate-800/60">
      <div className="flex items-center gap-2 mb-2.5">
        <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Demo Claims (Click to load):
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {DEMO_CLAIMS.map((item, idx) => (
          <button
            key={idx}
            type="button"
            disabled={disabled}
            onClick={() => onSelectClaim(item.text)}
            className="group inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-xs text-slate-300 hover:text-white transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed text-left focus:outline-none focus:ring-1 focus:ring-emerald-500"
          >
            <span className="text-[10px] font-semibold text-emerald-400/90 uppercase tracking-wide px-1.5 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40">
              {item.badge}
            </span>
            <span className="truncate max-w-[280px] sm:max-w-md font-mono text-[11px] text-slate-300 group-hover:text-slate-100">
              "{item.text}"
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
