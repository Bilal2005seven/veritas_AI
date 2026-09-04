import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, ShieldCheck, HelpCircle } from 'lucide-react';

const VERDICT_CONFIG = {
  VERIFIED: {
    title: 'VERIFIED',
    subtitle: 'Strong supporting evidence was found from credible sources.',
    badgeClass: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    cardClass: 'border-emerald-500/30 bg-gradient-to-b from-emerald-950/20 to-slate-900/40 shadow-emerald-950/20',
    icon: CheckCircle2,
    iconColor: 'text-emerald-400',
    accentColor: 'bg-emerald-500',
  },
  UNVERIFIED: {
    title: 'UNVERIFIED',
    subtitle: 'Available evidence is insufficient, mixed, or inconclusive.',
    badgeClass: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    cardClass: 'border-amber-500/30 bg-gradient-to-b from-amber-950/20 to-slate-900/40 shadow-amber-950/20',
    icon: AlertTriangle,
    iconColor: 'text-amber-400',
    accentColor: 'bg-amber-500',
  },
  CONTRADICTED: {
    title: 'CONTRADICTED',
    subtitle: 'Strong evidence actively refutes or contradicts the claim.',
    badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    cardClass: 'border-rose-500/30 bg-gradient-to-b from-rose-950/20 to-slate-900/40 shadow-rose-950/20',
    icon: XCircle,
    iconColor: 'text-rose-400',
    accentColor: 'bg-rose-500',
  },
};

export default function VerdictCard({ verdict, confidence, claim }) {
  const normVerdict = (verdict || 'UNVERIFIED').toUpperCase();
  const config = VERDICT_CONFIG[normVerdict] || VERDICT_CONFIG.UNVERIFIED;
  const Icon = config.icon;

  const formattedConfidence =
    typeof confidence === 'number'
      ? `${(confidence * 100).toFixed(1)}%`
      : 'N/A';

  return (
    <div className="space-y-4">
      {/* Verified Claim Quote Box */}
      <div className="p-4 sm:p-5 rounded-xl bg-slate-900/80 border border-slate-800">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-1.5 font-mono">
          Analyzed Claim
        </span>
        <p className="text-base sm:text-lg font-medium text-slate-100 font-sans italic">
          "{claim}"
        </p>
      </div>

      {/* Prominent Verdict Card */}
      <div
        className={`border rounded-2xl p-6 sm:p-8 backdrop-blur-sm shadow-xl relative overflow-hidden transition-all ${config.cardClass}`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          {/* Verdict Label & Description */}
          <div className="flex items-start gap-4">
            <div className={`p-3 rounded-2xl bg-slate-950/60 border border-slate-800 shadow-inner mt-0.5`}>
              <Icon className={`w-8 h-8 sm:w-10 sm:h-10 ${config.iconColor}`} />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Deterministic Verdict
                </span>
                <span
                  className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-full border uppercase tracking-wider ${config.badgeClass}`}
                >
                  {normVerdict}
                </span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-black tracking-tight text-white font-sans">
                {config.title}
              </h2>
              <p className="text-sm sm:text-base text-slate-300 mt-1 max-w-xl">
                {config.subtitle}
              </p>
            </div>
          </div>

          {/* Confidence Badge */}
          <div className="sm:text-right bg-slate-950/40 p-4 rounded-xl border border-slate-800/80 sm:min-w-[160px]">
            <div className="flex sm:justify-end items-center gap-1.5 text-xs text-slate-400 font-medium mb-1">
              <ShieldCheck className="w-3.5 h-3.5 text-slate-400" />
              <span>Verdict Confidence</span>
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold font-mono text-white">
              {formattedConfidence}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Source diversity calibrated
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
