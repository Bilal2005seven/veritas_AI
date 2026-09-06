import React from 'react';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HelpCircle,
  ShieldCheck,
  ExternalLink,
  Search,
  Flag,
  Sparkles,
  RotateCcw,
  BookOpen,
} from 'lucide-react';

const VERDICT_CONFIG = {
  SUPPORTED: {
    title: 'SUPPORTED',
    subtitle: 'Reliable available evidence generally confirms the core assertions of this claim.',
    badgeClass: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    cardClass: 'border-emerald-500/30 bg-gradient-to-b from-emerald-950/20 to-slate-900/40 shadow-emerald-950/20',
    icon: CheckCircle2,
    iconColor: 'text-emerald-400',
    dotColor: 'bg-emerald-400',
  },
  CONTRADICTED: {
    title: 'CONTRADICTED',
    subtitle: 'Reliable available evidence conflicts with or directly refutes the claim.',
    badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    cardClass: 'border-rose-500/30 bg-gradient-to-b from-rose-950/20 to-slate-900/40 shadow-rose-950/20',
    icon: XCircle,
    iconColor: 'text-rose-400',
    dotColor: 'bg-rose-400',
  },
  UNCERTAIN: {
    title: 'UNCERTAIN',
    subtitle: 'Evidence is insufficient, inconclusive, ambiguous, or in active dispute.',
    badgeClass: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    cardClass: 'border-amber-500/30 bg-gradient-to-b from-amber-950/20 to-slate-900/40 shadow-amber-950/20',
    icon: HelpCircle,
    iconColor: 'text-amber-400',
    dotColor: 'bg-amber-400',
  },
  MISLEADING: {
    title: 'MISLEADING',
    subtitle: 'The claim references real events or data, but omits critical context, exaggerates, or alters the timeline.',
    badgeClass: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    cardClass: 'border-purple-500/30 bg-gradient-to-b from-purple-950/20 to-slate-900/40 shadow-purple-950/20',
    icon: AlertTriangle,
    iconColor: 'text-purple-400',
    dotColor: 'bg-purple-400',
  },
};

export default function GeminiResult({ result, onReset }) {
  if (!result) return null;

  const {
    verdict = 'UNCERTAIN',
    confidence = 0.5,
    claim = '',
    summary = '',
    claims = [],
    supporting_evidence = [],
    contradicting_evidence = [],
    sources = [],
    red_flags = [],
    search_queries = [],
  } = result;

  const normVerdict = (verdict || 'UNCERTAIN').toUpperCase();
  const config = VERDICT_CONFIG[normVerdict] || VERDICT_CONFIG.UNCERTAIN;
  const Icon = config.icon;

  const formattedConfidence =
    typeof confidence === 'number'
      ? `${(confidence * 100).toFixed(0)}%`
      : 'N/A';

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header bar */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 relative">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${config.dotColor} opacity-75`} />
            <span className={`relative inline-flex rounded-full h-2 w-2 ${config.dotColor}`} />
          </span>
          <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 font-mono">
            Gemini Verification Report
          </span>
          <span className="text-slate-600">•</span>
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            Powered by Gemini 3.7 Flash & Google Search Grounding
          </span>
        </div>
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-cyan-400 transition-colors focus:outline-none"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>New Gemini verification</span>
          </button>
        )}
      </div>

      {/* 1. Analyzed Claim Box */}
      <div className="p-4 sm:p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-inner">
        <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-widest block mb-1.5 font-mono">
          Analyzed Claim / Subject
        </span>
        <p className="text-base sm:text-lg font-medium text-slate-100 font-sans italic">
          "{claim}"
        </p>
      </div>

      {/* 2. Prominent Verdict Card */}
      <div
        className={`border rounded-2xl p-6 sm:p-8 backdrop-blur-sm shadow-xl relative overflow-hidden transition-all ${config.cardClass}`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div className="flex items-start gap-4">
            <div className="p-3.5 rounded-2xl bg-slate-950/70 border border-slate-800 shadow-inner mt-0.5">
              <Icon className={`w-9 h-9 sm:w-11 sm:h-11 ${config.iconColor}`} />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-1.5">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Gemini Verdict
                </span>
                <span
                  className={`text-xs font-mono font-bold px-3 py-0.5 rounded-full border uppercase tracking-wider ${config.badgeClass}`}
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

          <div className="sm:text-right bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 sm:min-w-[190px]">
            <div className="flex sm:justify-end items-center gap-1.5 text-xs text-slate-400 font-medium mb-1">
              <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
              <span>Estimated Confidence</span>
            </div>
            <div className="text-3xl sm:text-4xl font-extrabold font-mono text-white">
              {formattedConfidence}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              Based on available evidence
            </p>
          </div>
        </div>
      </div>

      {/* 3. Evidence-based Summary */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-sm space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-cyan-400 uppercase tracking-wider font-mono">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span>Evidence-Based Synthesis</span>
        </div>
        <p className="text-slate-200 text-sm sm:text-base leading-relaxed">
          {summary}
        </p>
      </div>

      {/* 4. Sub-claims Breakdown (if present) */}
      {claims && claims.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 font-mono">
            Evaluated Sub-Claims ({claims.length})
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {claims.map((c, i) => (
              <div
                key={i}
                className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-2"
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border uppercase ${
                      VERDICT_CONFIG[c.verdict]?.badgeClass || 'bg-slate-800 text-slate-300'
                    }`}
                  >
                    {c.verdict}
                  </span>
                  {typeof c.confidence === 'number' && (
                    <span className="text-xs font-mono text-slate-400">
                      {(c.confidence * 100).toFixed(0)}% confidence
                    </span>
                  )}
                </div>
                <p className="text-sm font-medium text-slate-100">
                  {c.claim}
                </p>
                {c.explanation && (
                  <p className="text-xs text-slate-400 leading-normal">
                    {c.explanation}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. Supporting Evidence */}
      {supporting_evidence && supporting_evidence.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-emerald-400 font-mono">
              Supporting Evidence ({supporting_evidence.length})
            </h3>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {supporting_evidence.map((item, idx) => (
              <div
                key={idx}
                className="p-4 sm:p-5 rounded-xl border border-emerald-500/20 bg-emerald-950/10 hover:border-emerald-500/40 transition-all flex flex-col justify-between space-y-3"
              >
                <div className="space-y-2">
                  <span className="text-[11px] font-bold font-mono text-emerald-400 uppercase tracking-wide">
                    {item.source}
                  </span>
                  <h4 className="text-sm font-semibold text-slate-100 line-clamp-2">
                    {item.title}
                  </h4>
                  {item.snippet && (
                    <p className="text-xs text-slate-300 leading-relaxed italic border-l-2 border-emerald-500/40 pl-2">
                      "{item.snippet}"
                    </p>
                  )}
                </div>
                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400 hover:text-emerald-300 pt-1 transition-colors"
                  >
                    <span>View Source</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 6. Contradicting Evidence */}
      {contradicting_evidence && contradicting_evidence.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <XCircle className="w-4 h-4 text-rose-400" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-rose-400 font-mono">
              Contradicting Evidence ({contradicting_evidence.length})
            </h3>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {contradicting_evidence.map((item, idx) => (
              <div
                key={idx}
                className="p-4 sm:p-5 rounded-xl border border-rose-500/20 bg-rose-950/10 hover:border-rose-500/40 transition-all flex flex-col justify-between space-y-3"
              >
                <div className="space-y-2">
                  <span className="text-[11px] font-bold font-mono text-rose-400 uppercase tracking-wide">
                    {item.source}
                  </span>
                  <h4 className="text-sm font-semibold text-slate-100 line-clamp-2">
                    {item.title}
                  </h4>
                  {item.snippet && (
                    <p className="text-xs text-slate-300 leading-relaxed italic border-l-2 border-rose-500/40 pl-2">
                      "{item.snippet}"
                    </p>
                  )}
                </div>
                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-rose-400 hover:text-rose-300 pt-1 transition-colors"
                  >
                    <span>View Source</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 7. Red Flags */}
      {red_flags && red_flags.length > 0 && (
        <div className="p-5 rounded-xl bg-purple-950/10 border border-purple-500/20 space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-purple-400 uppercase tracking-wider font-mono">
            <Flag className="w-4 h-4" />
            <span>Detected Warning Signs & Missing Context ({red_flags.length})</span>
          </div>
          <ul className="space-y-2 text-sm text-slate-300">
            {red_flags.map((flag, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-purple-400 font-bold">•</span>
                <span>{flag}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 8. Web Search Grounding Queries */}
      {search_queries && search_queries.length > 0 && (
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">
            <Search className="w-3.5 h-3.5 text-cyan-400" />
            <span>Web Search Grounding Queries ({search_queries.length})</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {search_queries.map((q, idx) => (
              <span
                key={idx}
                className="text-xs font-mono bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800 text-slate-300"
              >
                "{q}"
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 9. All Sources Cited */}
      {sources && sources.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-cyan-400 font-mono">
              Verified Web Sources ({sources.length})
            </h3>
          </div>
          <div className="grid gap-2.5 sm:grid-cols-2">
            {sources.map((s, idx) => (
              <a
                key={idx}
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 hover:border-cyan-500/40 transition-all flex items-center justify-between group"
              >
                <div className="min-w-0 pr-2">
                  <p className="text-xs font-medium text-slate-200 group-hover:text-cyan-400 transition-colors truncate">
                    {s.title || s.source}
                  </p>
                  <p className="text-[10px] text-slate-500 font-mono truncate">
                    {s.source}
                  </p>
                </div>
                <ExternalLink className="w-3.5 h-3.5 text-slate-500 group-hover:text-cyan-400 shrink-0 transition-colors" />
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
