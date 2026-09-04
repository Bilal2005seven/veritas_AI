import React from 'react';
import {
  TrendingUp,
  TrendingDown,
  Award,
  Globe2,
  FileCheck,
  BarChart3,
} from 'lucide-react';

function formatPct(val) {
  if (typeof val !== 'number' || isNaN(val)) return '—';
  return `${(val * 100).toFixed(1)}%`;
}

export default function ScoreCards({
  supportScore,
  contradictionScore,
  credibility,
  evidence = [],
  sources = [],
}) {
  const comp = credibility?.component_scores || {};

  // Resolved scores (prefer top-level, fallback to component_scores)
  const resolvedSupport =
    typeof supportScore === 'number' ? supportScore : comp.support_score;
  const resolvedContradiction =
    typeof contradictionScore === 'number'
      ? contradictionScore
      : comp.contradiction_score;
  const evidenceQuality = comp.evidence_quality_score;
  const sourceQuality = comp.source_quality_score;
  const overallCredibility = credibility?.overall_score;

  // Evidence counts
  const supportingCount = evidence.filter(
    (e) => e.nli_label === 'ENTAILMENT'
  ).length;
  const contradictingCount = evidence.filter(
    (e) => e.nli_label === 'CONTRADICTION'
  ).length;
  const neutralCount = evidence.filter(
    (e) => e.nli_label !== 'ENTAILMENT' && e.nli_label !== 'CONTRADICTION'
  ).length;

  const uniqueDomainsCount = sources.length > 0 ? sources.length : new Set(evidence.map(e => e.source_name).filter(Boolean)).size;

  return (
    <div className="space-y-6">
      {/* Metrics Grid */}
      <div>
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-emerald-400" />
          Credibility & Stance Signals
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          {/* Support Score */}
          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
              <span className="font-medium">Support Signal</span>
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-emerald-400">
              {formatPct(resolvedSupport)}
            </div>
            {/* Meter */}
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(100, Math.max(0, (resolvedSupport || 0) * 100))}%`,
                }}
              />
            </div>
          </div>

          {/* Contradiction Score */}
          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
              <span className="font-medium">Contradiction Signal</span>
              <TrendingDown className="w-4 h-4 text-rose-400" />
            </div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-rose-400">
              {formatPct(resolvedContradiction)}
            </div>
            {/* Meter */}
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-rose-500 h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(
                    100,
                    Math.max(0, (resolvedContradiction || 0) * 100)
                  )}%`,
                }}
              />
            </div>
          </div>

          {/* Evidence Quality Score */}
          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
              <span className="font-medium">Evidence Quality</span>
              <FileCheck className="w-4 h-4 text-teal-400" />
            </div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-100">
              {formatPct(evidenceQuality)}
            </div>
            {/* Meter */}
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-teal-500 h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(100, Math.max(0, (evidenceQuality || 0) * 100))}%`,
                }}
              />
            </div>
          </div>

          {/* Source Quality Score */}
          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
              <span className="font-medium">Source Authority</span>
              <Award className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-100">
              {formatPct(sourceQuality)}
            </div>
            {/* Meter */}
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div
                className="bg-indigo-500 h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(100, Math.max(0, (sourceQuality || 0) * 100))}%`,
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Compact Source Overview Bar */}
      <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-2 text-slate-400">
          <Globe2 className="w-4 h-4 text-slate-400" />
          <span className="font-medium text-slate-300">Evidence Distribution:</span>
        </div>

        <div className="flex flex-wrap items-center gap-3 sm:gap-6 font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-slate-400" />
            <span className="text-slate-400">Total Analyzed:</span>
            <span className="font-bold text-white">{evidence.length}</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-slate-400">Supporting:</span>
            <span className="font-bold text-emerald-400">{supportingCount}</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-400" />
            <span className="text-slate-400">Contradicting:</span>
            <span className="font-bold text-rose-400">{contradictingCount}</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <span className="text-slate-400">Neutral:</span>
            <span className="font-bold text-slate-300">{neutralCount}</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-400" />
            <span className="text-slate-400">Unique Sources:</span>
            <span className="font-bold text-indigo-300">{uniqueDomainsCount}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
