/**
 * VeritasAI Frontend — API Service Layer
 * External interface to the frozen FastAPI backend.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

/**
 * Verify a claim through the backend verification pipeline.
 *
 * @param {string} claim The factual claim text to verify.
 * @param {AbortSignal} [signal] Optional abort signal for cancellation.
 * @returns {Promise<import('./types').VerifyResponse>}
 */
export async function verifyClaim(claim, signal) {
  const trimmedClaim = (claim || '').trim();

  if (!trimmedClaim) {
    throw new Error('Please enter a claim before verifying.');
  }

  const endpoint = `${BASE_URL}/api/v1/verify`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ claim: trimmedClaim }),
      signal,
    });

    if (response.status === 422) {
      const errData = await response.json().catch(() => ({}));
      const detail = errData.detail || 'Claim must not be empty or whitespace-only.';
      throw new Error(typeof detail === 'string' ? detail : 'Invalid claim format.');
    }

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('The verification request was cancelled.');
    }

    // Network error (e.g. Failed to fetch, Connection Refused)
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.message.includes('Load failed')) {
      throw new Error('Unable to connect to the verification engine. Ensure the FastAPI backend is running.');
    }

    throw err;
  }
}

/**
 * Verify a claim, article URL, or screenshot using Gemini 3.7 Flash.
 *
 * @param {Object} params
 * @param {string} [params.claim] Optional text claim.
 * @param {string} [params.url] Optional public news URL.
 * @param {File} [params.image] Optional image/screenshot File.
 * @param {AbortSignal} [signal] Optional cancellation signal.
 * @returns {Promise<any>}
 */
export async function verifyWithGemini({ claim, url, image }, signal) {
  if (!claim && !url && !image) {
    throw new Error('Please provide at least a claim, news URL, or screenshot to verify.');
  }

  const formData = new FormData();
  if (claim && claim.trim()) {
    formData.append('claim', claim.trim());
  }
  if (url && url.trim()) {
    formData.append('url', url.trim());
  }
  if (image) {
    formData.append('image', image);
  }

  const endpoint = `${BASE_URL}/api/v1/gemini-verify`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        // Do NOT manually set Content-Type so browser sets boundary automatically
      },
      body: formData,
      signal,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const detail = errData.detail || `Server returned status ${response.status}`;
      throw new Error(typeof detail === 'string' ? detail : 'Gemini verification failed.');
    }

    const data = await response.json();
    return data;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('The verification request was cancelled.');
    }

    if (
      err.message.includes('Failed to fetch') ||
      err.message.includes('NetworkError') ||
      err.message.includes('Load failed')
    ) {
      throw new Error(
        'Unable to connect to the verification engine. Ensure the FastAPI backend is running.'
      );
    }

    throw err;
  }
}

/**
 * Probe the backend liveness endpoint.
 *
 * @returns {Promise<{ status: string, version: string }>}
 */
export async function checkHealth() {
  const endpoint = `${BASE_URL}/health`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      return await response.json();
    }
    return { status: 'error', version: 'unknown' };
  } catch {
    clearTimeout(timeoutId);
    return { status: 'offline', version: 'unknown' };
  }
}
