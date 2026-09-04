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
import { verifyClaim } from './services/api';
import { RotateCcw, Shield } from 'lucide-react';

export default function App() {
  const [claim, setClaim] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const resultRef = useRef(null);

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

      // Auto-scroll to results smoothly
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

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-10">
        {/* Claim Input Hero Section */}
        <section aria-label="Claim submission">
          <ClaimInput
            claim={claim}
            setClaim={(c) => {
              setClaim(c);
              if (error) setError(null);
            }}
            onVerify={() => handleVerify(claim)}
            isLoading={isLoading}
          />
        </section>

        {/* Dynamic Display Area */}
        <section ref={resultRef} aria-label="Verification results" className="scroll-mt-24 space-y-8">
          {isLoading && <LoadingState />}

          {error && !isLoading && (
            <ErrorState message={error} onRetry={() => handleVerify(claim)} />
          )}

          {result && !isLoading && (
            <div className="space-y-8 animate-fade-in">
              {/* Reset / New Verification Action */}
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
                  Verification Report
                </span>
                <button
                  type="button"
                  onClick={handleReset}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-emerald-400 transition-colors focus:outline-none"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Verify another claim</span>
                </button>
              </div>

              {/* 1. Verdict & Claim Card */}
              <VerdictCard
                verdict={result.verdict}
                confidence={result.confidence}
                claim={result.claim}
              />

              {/* 2. AI Synthesized Explanation */}
              <ExplanationCard explanation={result.explanation} />

              {/* 3. Credibility & Score Section */}
              <ScoreCards
                supportScore={result.support_score}
                contradictionScore={result.contradiction_score}
                credibility={result.credibility}
                evidence={result.evidence}
                sources={result.sources}
              />

              {/* 4. Evidence Breakdown (Supporting, Contradicting, Neutral) */}
              <EvidenceSection evidence={result.evidence} />
            </div>
          )}

          {!result && !isLoading && !error && <EmptyState />}
        </section>
      </main>

      {/* Clean Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-emerald-500" />
            <span className="font-semibold text-slate-400">VeritasAI V1</span>
            <span>—</span>
            <span>Evidence-backed claim verification system</span>
          </div>
          <div className="text-[11px] text-slate-600 font-mono">
            Powered by MiniLM Embeddings & NLI Cross-Encoder
          </div>
        </div>
      </footer>
    </div>
  );
}
