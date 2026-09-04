import React, { useRef } from 'react';
import { Search, Loader2, CornerDownLeft, XCircle } from 'lucide-react';
import DemoClaims from './DemoClaims';

export default function ClaimInput({
  claim,
  setClaim,
  onVerify,
  isLoading,
}) {
  const textareaRef = useRef(null);

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isLoading && claim.trim().length > 0) {
        onVerify();
      }
    }
  };

  const handleClear = () => {
    setClaim('');
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const isButtonDisabled = isLoading || !claim.trim();

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xl backdrop-blur-sm relative overflow-hidden">
      {/* Subtle decorative glow */}
      <div className="absolute -top-24 -right-24 w-72 h-72 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-3xl">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white font-sans mb-2">
          Verify a Claim
        </h1>
        <p className="text-sm sm:text-base text-slate-400 mb-6">
          Enter any statement or factual claim. VeritasAI will retrieve live web evidence,
          evaluate semantic similarity, cross-encode NLI relationships, and calculate credibility.
        </p>

        {/* Textarea container */}
        <div className="relative group">
          <textarea
            ref={textareaRef}
            rows={3}
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder="Example: The Earth revolves around the Sun."
            className="w-full bg-slate-950/90 border border-slate-700/80 group-hover:border-slate-600 rounded-xl px-4 py-3.5 text-slate-100 placeholder:text-slate-500 text-sm sm:text-base focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-colors resize-none disabled:opacity-60 disabled:cursor-not-allowed shadow-inner"
          />

          {claim && !isLoading && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute right-3 top-3 text-slate-500 hover:text-slate-300 transition-colors p-1"
              title="Clear text"
            >
              <XCircle className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Action bar */}
        <div className="mt-4 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          <span className="text-xs text-slate-500 hidden sm:flex items-center gap-1">
            Tip: Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px]">Ctrl</kbd> + <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px]">Enter</kbd> to verify
          </span>

          <button
            type="button"
            disabled={isButtonDisabled}
            onClick={() => onVerify()}
            className={`inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200 shadow-md ${
              isButtonDisabled
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-800'
                : 'bg-emerald-500 hover:bg-emerald-400 active:bg-emerald-600 text-slate-950 shadow-emerald-950/40 hover:shadow-emerald-500/20'
            }`}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
                <span>Verifying Claim...</span>
              </>
            ) : (
              <>
                <Search className="w-4 h-4 text-slate-950" />
                <span>Verify Claim</span>
                <CornerDownLeft className="w-3.5 h-3.5 text-slate-900 hidden sm:inline" />
              </>
            )}
          </button>
        </div>

        {/* Demo claims */}
        <DemoClaims onSelectClaim={(c) => setClaim(c)} disabled={isLoading} />
      </div>
    </div>
  );
}
