import React, { useState } from 'react';
import {
  CheckCircle2,
  XCircle,
  MinusCircle,
  ChevronDown,
  ChevronUp,
  Layers,
} from 'lucide-react';
import EvidenceCard from './EvidenceCard';

export default function EvidenceSection({ evidence = [] }) {
  const [isNeutralExpanded, setIsNeutralExpanded] = useState(false);

  // Group evidence by NLI label
  const supporting = evidence.filter((e) => e.nli_label === 'ENTAILMENT');
  const contradicting = evidence.filter((e) => e.nli_label === 'CONTRADICTION');
  const neutral = evidence.filter(
    (e) => e.nli_label !== 'ENTAILMENT' && e.nli_label !== 'CONTRADICTION'
  );

  return (
    <div className="space-y-8">
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg sm:text-xl font-bold text-white flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-400" />
            Extracted Web Evidence
          </h3>
          <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
            Classified via cross-encoder NLI and evaluated for topical relevance.
          </p>
        </div>
        <span className="text-xs font-mono text-slate-400 bg-slate-900 px-3 py-1 rounded-full border border-slate-800">
          {evidence.length} items total
        </span>
      </div>

      {/* A. Supporting Evidence */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <h4 className="text-sm font-bold text-emerald-400 uppercase tracking-wider">
              Supporting Evidence ({supporting.length})
            </h4>
          </div>
          <span className="text-xs text-slate-500">
            Passes NLI ENTAILMENT threshold
          </span>
        </div>

        {supporting.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {supporting.map((item, idx) => (
              <EvidenceCard key={idx} item={item} variant="supporting" />
            ))}
          </div>
        ) : (
          <div className="p-4 rounded-xl border border-slate-800/80 bg-slate-950/40 text-xs text-slate-500 italic">
            No direct supporting evidence found meeting the required semantic threshold.
          </div>
        )}
      </div>

      {/* B. Contradicting Evidence */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <XCircle className="w-4 h-4 text-rose-400" />
            <h4 className="text-sm font-bold text-rose-400 uppercase tracking-wider">
              Contradicting Evidence ({contradicting.length})
            </h4>
          </div>
          <span className="text-xs text-slate-500">
            Passes NLI CONTRADICTION threshold
          </span>
        </div>

        {contradicting.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {contradicting.map((item, idx) => (
              <EvidenceCard key={idx} item={item} variant="contradicting" />
            ))}
          </div>
        ) : (
          <div className="p-4 rounded-xl border border-slate-800/80 bg-slate-950/40 text-xs text-slate-500 italic">
            No direct contradictory evidence surfaced for this claim.
          </div>
        )}
      </div>

      {/* C. Neutral Evidence (Collapsible) */}
      <div className="space-y-3 pt-2">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => setIsNeutralExpanded(!isNeutralExpanded)}
            className="flex items-center gap-2 text-sm font-bold text-slate-400 hover:text-slate-200 transition-colors uppercase tracking-wider group focus:outline-none"
          >
            <MinusCircle className="w-4 h-4 text-slate-500 group-hover:text-slate-300" />
            <span>Neutral / Inconclusive Evidence ({neutral.length})</span>
            {isNeutralExpanded ? (
              <ChevronUp className="w-4 h-4 text-slate-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-500" />
            )}
          </button>
          <span className="text-xs text-slate-500 hidden sm:inline">
            Does not directly refute or prove the claim
          </span>
        </div>

        {isNeutralExpanded && (
          <div className="space-y-3 animate-fade-in">
            <p className="text-xs text-slate-500">
              These articles were retrieved due to topical similarity but either do not directly address the claim, are neutral questions/reports, or did not pass the decisive stance threshold.
            </p>
            {neutral.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {neutral.map((item, idx) => (
                  <EvidenceCard key={idx} item={item} variant="neutral" />
                ))}
              </div>
            ) : (
              <div className="p-4 rounded-xl border border-slate-800/80 bg-slate-950/40 text-xs text-slate-500 italic">
                No neutral evidence items.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
