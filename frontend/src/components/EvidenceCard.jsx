import React from 'react';
import { ExternalLink, Calendar, CheckCircle2, XCircle, MinusCircle } from 'lucide-react';

export default function EvidenceCard({ item, variant = 'neutral' }) {
  const {
    title,
    url,
    source_name,
    relevance_score,
    nli_label,
    entailment_score,
    contradiction_score,
    neutral_score,
    published_at,
    content_snippet,
  } = item;

  const relevancePct =
    typeof relevance_score === 'number'
      ? `${(relevance_score * 100).toFixed(0)}%`
      : '—';

  // Variant styling
  const isSupporting = variant === 'supporting' || nli_label === 'ENTAILMENT';
  const isContradicting =
    variant === 'contradicting' || nli_label === 'CONTRADICTION';

  let borderClass = 'border-slate-800 hover:border-slate-700 bg-slate-900/60';
  let badgeClass = 'bg-slate-800 text-slate-300 border-slate-700';
  let StatusIcon = MinusCircle;
  let iconColor = 'text-slate-400';

  if (isSupporting) {
    borderClass =
      'border-emerald-500/20 hover:border-emerald-500/40 bg-emerald-950/10';
    badgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    StatusIcon = CheckCircle2;
    iconColor = 'text-emerald-400';
  } else if (isContradicting) {
    borderClass =
      'border-rose-500/20 hover:border-rose-500/40 bg-rose-950/10';
    badgeClass = 'bg-rose-500/10 text-rose-400 border-rose-500/30';
    StatusIcon = XCircle;
    iconColor = 'text-rose-400';
  }

  // Format date if present
  let formattedDate = null;
  if (published_at) {
    try {
      const d = new Date(published_at);
      if (!isNaN(d.getTime())) {
        formattedDate = d.toLocaleDateString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
        });
      }
    } catch {
      formattedDate = published_at;
    }
  }

  return (
    <div
      className={`border rounded-xl p-4 sm:p-5 transition-all duration-200 flex flex-col justify-between ${borderClass}`}
    >
      <div>
        {/* Top bar: Source + Date + Stance Badge */}
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-mono text-xs font-semibold text-slate-300 truncate max-w-[200px]">
              {source_name || (url ? new URL(url).hostname : 'Web Source')}
            </span>
            {formattedDate && (
              <span className="text-[11px] text-slate-500 flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {formattedDate}
              </span>
            )}
          </div>

          <div
            className={`inline-flex items-center gap-1 text-[11px] font-mono font-bold px-2 py-0.5 rounded border ${badgeClass}`}
          >
            <StatusIcon className={`w-3 h-3 ${iconColor}`} />
            <span>{nli_label || 'NEUTRAL'}</span>
          </div>
        </div>

        {/* Article Title */}
        <h4 className="text-sm sm:text-base font-semibold text-slate-100 mb-2 leading-snug line-clamp-2">
          {title || 'Untitled Evidence Source'}
        </h4>

        {/* Content Snippet */}
        {content_snippet && (
          <p className="text-xs sm:text-sm text-slate-400 mb-4 line-clamp-3 leading-relaxed">
            "{content_snippet}"
          </p>
        )}
      </div>

      {/* Footer / Signals */}
      <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between gap-3 text-xs mt-2">
        <div className="flex items-center gap-3">
          <div>
            <span className="text-slate-500 text-[11px]">Relevance: </span>
            <span className="font-mono font-bold text-slate-200">
              {relevancePct}
            </span>
          </div>

          {/* Probability indicators */}
          {typeof entailment_score === 'number' &&
            typeof contradiction_score === 'number' && (
              <div className="hidden sm:flex items-center gap-2 text-[11px] font-mono text-slate-500">
                <span title="Entailment probability">
                  ENT: {(entailment_score * 100).toFixed(0)}%
                </span>
                <span title="Contradiction probability">
                  CON: {(contradiction_score * 100).toFixed(0)}%
                </span>
              </div>
            )}
        </div>

        {url ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400 hover:text-emerald-300 hover:underline transition-colors group"
          >
            <span>View Source</span>
            <ExternalLink className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
          </a>
        ) : (
          <span className="text-slate-600 text-[11px]">No link available</span>
        )}
      </div>
    </div>
  );
}
