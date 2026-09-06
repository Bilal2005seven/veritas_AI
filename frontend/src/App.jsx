import React, { useState, useRef } from 'react';
import Navbar from './components/Navbar';
import ClaimInput from './components/ClaimInput';
import LoadingState from './components/LoadingState';
import VerdictCard from './components/VerdictCard';
import ScoreCards from './components/ScoreCards';
import EvidenceSection from './components/EvidenceSection';
import ExplanationCard from './components/ExplanationCard';
import EmptyState from './components/EmptyState';
import ErrorState from './components/ErrorState';
import GeminiVerification from './components/GeminiVerification';
import GeminiLoadingState from './components/GeminiLoadingState';
import GeminiResult from './components/GeminiResult';
import { verifyClaim, verifyWithGemini } from './services/api';
import { RotateCcw, Shield, Sparkles, FileText } from 'lucide-react';

export default function App() {
  // Existing Text Verification Pipeline State
  const [claim, setClaim] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // New Gemini Verification Pipeline State (Completely Independent)
  const [geminiLoading, setGeminiLoading] = useState(false);
  const [geminiError, setGeminiError] = useState(null);
  const [geminiResult, setGeminiResult] = useState(null);
  const [lastGeminiParams, setLastGeminiParams] = useState(null);

  const resultRef = useRef(null);
  const geminiResultRef = useRef(null);

  // Existing text claim verification handler
  const handleVerify = async (overrideClaim) => {
    const claimToVerify = (overrideClaim || claim).trim();
    if (!claimToVerify) {
      setError('Please enter a claim before verifying.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const data = await verifyClaim(claimToVerify);
      setResult(data);

      setTimeout(() => {
        if (resultRef.current) {
          resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (err) {
      console.error('Verification failure:', err);
      setError(err.message || 'Something went wrong while verifying this claim.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setClaim('');
    setResult(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // New Gemini verification handler
  const handleGeminiVerify = async (params) => {
    setLastGeminiParams(params);
    setGeminiLoading(true);
    setGeminiError(null);

    try {
      const data = await verifyWithGemini(params);
      setGeminiResult(data);

      setTimeout(() => {
        if (geminiResultRef.current) {
          geminiResultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (err) {
      console.error('Gemini verification failure:', err);
      setGeminiError(err.message || 'Gemini verification failed. Please check inputs or try again.');
    } finally {
      setGeminiLoading(false);
    }
  };

  const handleGeminiReset = () => {
    setGeminiResult(null);
    setGeminiError(null);
  };

  const hasAnyResults = result || geminiResult;
  const isAnyLoading = isLoading || geminiLoading;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-14">
        {/* ============================================================ */}
        {/* SECTION 1: EXISTING TEXT CLAIM VERIFICATION (NLI + RAG)     */}
        {/* ============================================================ */}
        <section aria-label="Text claim submission" className="space-y-4">
          <div className="flex items-center gap-2 pb-1 border-b border-slate-800/80">
            <FileText className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 font-mono">
              Section 1 — Text Claim Verification
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              NLI + RAG + Credibility
            </span>
          </div>

          <ClaimInput
            claim={claim}
            setClaim={(c) => {
              setClaim(c);
              if (error) setError(null);
            }}
            onVerify={() => handleVerify(claim)}
            isLoading={isLoading}
          />

          {/* Section 1 Results & State */}
          <div ref={resultRef} className="scroll-mt-24 space-y-8">
            {isLoading && <LoadingState />}

            {error && !isLoading && (
              <ErrorState message={error} onRetry={() => handleVerify(claim)} />
            )}

            {result && !isLoading && (
              <div className="space-y-8 animate-fade-in mt-6">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
                    VeritasAI Pipeline Report
                  </span>
                  <button
                    type="button"
                    onClick={handleReset}
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-emerald-400 transition-colors focus:outline-none"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Clear text report</span>
                  </button>
                </div>

                <VerdictCard
                  verdict={result.verdict}
                  confidence={result.confidence}
                  claim={result.claim}
                />

                <ExplanationCard explanation={result.explanation} />

                <ScoreCards
                  supportScore={result.support_score}
                  contradictionScore={result.contradiction_score}
                  credibility={result.credibility}
                  evidence={result.evidence}
                  sources={result.sources}
                />

                <EvidenceSection evidence={result.evidence} />
              </div>
            )}
          </div>
        </section>

        {/* Visual Divider Between Engines */}
        <div className="relative flex py-2 items-center">
          <div className="flex-grow border-t border-slate-800/80"></div>
          <span className="flex-shrink mx-4 px-3 py-1 rounded-full text-xs font-mono font-semibold uppercase tracking-widest text-slate-500 bg-slate-900 border border-slate-800">
            or choose multimodal verification
          </span>
          <div className="flex-grow border-t border-slate-800/80"></div>
        </div>

        {/* ============================================================ */}
        {/* SECTION 2: NEW GEMINI VERIFICATION (URL + SCREENSHOT)       */}
        {/* ============================================================ */}
        <section aria-label="Gemini Multimodal Verification" className="space-y-8">
          <GeminiVerification
            onVerify={handleGeminiVerify}
            isLoading={geminiLoading}
          />

          {/* Section 2 Results & State */}
          <div ref={geminiResultRef} className="scroll-mt-24 space-y-8">
            {geminiLoading && <GeminiLoadingState />}

            {geminiError && !geminiLoading && (
              <ErrorState
                message={geminiError}
                onRetry={() => lastGeminiParams && handleGeminiVerify(lastGeminiParams)}
              />
            )}

            {geminiResult && !geminiLoading && (
              <GeminiResult
                result={geminiResult}
                onReset={handleGeminiReset}
              />
            )}
          </div>
        </section>

        {/* Empty State when no verification has been performed yet */}
        {!hasAnyResults && !isAnyLoading && !error && !geminiError && (
          <EmptyState />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-emerald-500" />
            <span className="font-semibold text-slate-400">VeritasAI V1</span>
            <span>—</span>
            <span>Dual-Engine Verification Architecture</span>
          </div>
          <div className="text-[11px] text-slate-600 font-mono flex items-center gap-2">
            <span>Engine 1: MiniLM + NLI Cross-Encoder</span>
            <span>•</span>
            <span className="text-cyan-500/80">Engine 2: Gemini 3.7 Flash & Google Grounding</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

