import React, { useState, useRef } from 'react';
import {
  Sparkles,
  Link as LinkIcon,
  Image as ImageIcon,
  Upload,
  X,
  FileCheck,
  AlertCircle,
  ArrowRight,
} from 'lucide-react';

export default function GeminiVerification({ onVerify, isLoading }) {
  // URL Section State
  const [url, setUrl] = useState('');
  const [urlClaim, setUrlClaim] = useState('');
  const [urlError, setUrlError] = useState(null);

  // Image Section State
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageClaim, setImageClaim] = useState('');
  const [imageError, setImageError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

  // URL Submission Handler
  const handleVerifyUrl = (e) => {
    e.preventDefault();
    setUrlError(null);

    const trimmedUrl = url.trim();
    if (!trimmedUrl) {
      setUrlError('Please enter a news or article URL.');
      return;
    }

    if (!trimmedUrl.startsWith('http://') && !trimmedUrl.startsWith('https://')) {
      setUrlError('Please enter a valid public URL starting with http:// or https://');
      return;
    }

    onVerify({
      url: trimmedUrl,
      claim: urlClaim.trim() || undefined,
    });
  };

  // Image Selection Handler
  const handleFileSelect = (file) => {
    setImageError(null);
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      setImageError('Please upload a valid JPG, PNG, or WEBP image.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setImageError('File size exceeds 10MB limit.');
      return;
    }

    setImageFile(file);
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setImageFile(null);
    setImagePreview(null);
    setImageError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleVerifyImage = (e) => {
    e.preventDefault();
    setImageError(null);

    if (!imageFile) {
      setImageError('Please choose or drop a screenshot image to verify.');
      return;
    }

    onVerify({
      image: imageFile,
      claim: imageClaim.trim() || undefined,
    });
  };

  return (
    <section aria-label="Gemini Verification" className="space-y-6">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg sm:text-xl font-bold tracking-tight text-white font-sans">
              Gemini Verification
            </h2>
            <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 uppercase tracking-wider">
              Multimodal
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Verify modern web articles and screenshots using Google Search grounding and Gemini 3.7 Flash.
          </p>
        </div>

        <div className="flex items-center gap-2 text-[11px] text-slate-500 font-mono">
          <span className="hidden sm:inline">Powered by Gemini 3.7 Flash</span>
          <span className="hidden sm:inline">•</span>
          <span>Web-grounded assessment</span>
        </div>
      </div>

      {/* Two Column Grid for Desktop */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CARD 1: Verify a News URL */}
        <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 hover:border-slate-700/80 transition-all flex flex-col justify-between space-y-5">
          <div className="space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                <LinkIcon className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-100 font-sans">
                  Verify a News URL
                </h3>
                <p className="text-xs text-slate-400">
                  Analyze public articles with live web grounding
                </p>
              </div>
            </div>

            <form onSubmit={handleVerifyUrl} className="space-y-3.5">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1 font-mono">
                  Article URL <span className="text-rose-400">*</span>
                </label>
                <div className="relative">
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value);
                      if (urlError) setUrlError(null);
                    }}
                    placeholder="https://example.com/news/article"
                    disabled={isLoading}
                    className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950/80 border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 text-sm text-slate-100 placeholder-slate-600 outline-none font-mono transition-all"
                  />
                </div>
                {urlError && (
                  <p className="flex items-center gap-1.5 text-xs text-rose-400 mt-1.5 font-medium">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{urlError}</span>
                  </p>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1 font-mono">
                  Specific Claim / Question <span className="text-slate-500 font-normal">(Optional)</span>
                </label>
                <input
                  type="text"
                  value={urlClaim}
                  onChange={(e) => setUrlClaim(e.target.value)}
                  placeholder="e.g. Did PM Modi celebrate the GDP report?"
                  disabled={isLoading}
                  className="w-full px-3.5 py-2 rounded-xl bg-slate-950/80 border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 text-xs text-slate-200 placeholder-slate-600 outline-none transition-all"
                />
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full mt-2 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-sm shadow-md shadow-cyan-950/40 transition-all disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-cyan-400"
              >
                <span>Verify URL</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>
          </div>

          <div className="pt-2 border-t border-slate-800/60 text-[11px] text-slate-500 font-mono">
            Directly extracts factual claims and retrieves corroborating evidence.
          </div>
        </div>

        {/* CARD 2: Verify a Screenshot */}
        <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 hover:border-slate-700/80 transition-all flex flex-col justify-between space-y-5">
          <div className="space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
                <ImageIcon className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-100 font-sans">
                  Verify a Screenshot
                </h3>
                <p className="text-xs text-slate-400">
                  Multimodal vision analysis of social posts or headlines
                </p>
              </div>
            </div>

            <form onSubmit={handleVerifyImage} className="space-y-3.5">
              {/* Drop Zone */}
              {!imagePreview ? (
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    if (e.dataTransfer.files?.[0]) {
                      handleFileSelect(e.dataTransfer.files[0]);
                    }
                  }}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
                    isDragging
                      ? 'border-purple-500 bg-purple-950/20'
                      : 'border-slate-800 hover:border-purple-500/40 bg-slate-950/40'
                  }`}
                >
                  <Upload className="w-6 h-6 text-purple-400 mx-auto mb-1.5" />
                  <p className="text-xs font-medium text-slate-200">
                    Drop screenshot here or <span className="text-purple-400 underline">browse</span>
                  </p>
                  <p className="text-[10px] text-slate-500 mt-1 font-mono">
                    JPG, PNG, or WEBP (up to 10MB)
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/jpg"
                    onChange={(e) => handleFileSelect(e.target.files?.[0])}
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="p-3 rounded-xl bg-slate-950/80 border border-purple-500/30 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <img
                      src={imagePreview}
                      alt="Upload preview"
                      className="w-12 h-12 object-cover rounded-lg border border-slate-800"
                    />
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-slate-200 truncate">
                        {imageFile.name}
                      </p>
                      <p className="text-[10px] text-slate-500 font-mono">
                        {(imageFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={handleRemoveImage}
                    disabled={isLoading}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-900 transition-colors"
                    title="Remove image"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              )}

              {imageError && (
                <p className="flex items-center gap-1.5 text-xs text-rose-400 font-medium">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{imageError}</span>
                </p>
              )}

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1 font-mono">
                  Additional Note <span className="text-slate-500 font-normal">(Optional)</span>
                </label>
                <input
                  type="text"
                  value={imageClaim}
                  onChange={(e) => setImageClaim(e.target.value)}
                  placeholder="e.g. Is this breaking news banner real?"
                  disabled={isLoading}
                  className="w-full px-3.5 py-2 rounded-xl bg-slate-950/80 border border-slate-800 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 text-xs text-slate-200 placeholder-slate-600 outline-none transition-all"
                />
              </div>

              <button
                type="submit"
                disabled={isLoading || !imageFile}
                className="w-full mt-2 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-sm shadow-md shadow-purple-950/40 transition-all disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-purple-400"
              >
                <span>Verify Image</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>
          </div>

          <div className="pt-2 border-t border-slate-800/60 text-[11px] text-slate-500 font-mono">
            Gemini directly reads headlines, dates, tickers, and images.
          </div>
        </div>
      </div>
    </section>
  );
}
