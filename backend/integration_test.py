"""
Real end-to-end integration test script for VeritasAI V1.
Sends live POST /api/v1/verify requests to the running server and prints
a structured report for each claim. ASCII-only output for Windows cp1252.
"""
import json
import sys
import urllib.request
import urllib.error

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

CLAIMS = [
    "The Earth revolves around the Sun.",
    "Kal Damoh ke post office ke paas do cars ka accident hua tha.",
]

def post_verify(claim: str):
    payload = json.dumps({"claim": claim}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/verify",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        status = resp.status
        body = json.loads(resp.read().decode())
    return status, body

def report(claim: str, status: int, data: dict):
    ev = data.get("evidence", [])
    supporting    = [e for e in ev if e.get("nli_label") == "ENTAILMENT"]
    contradicting = [e for e in ev if e.get("nli_label") == "CONTRADICTION"]
    neutral       = [e for e in ev if e.get("nli_label") not in ("ENTAILMENT", "CONTRADICTION")]

    cred = data.get("credibility") or {}
    comp = cred.get("component_scores", {})

    sep = "=" * 72
    line = "-" * 72

    print(sep)
    print(f"CLAIM : {claim}")
    print(sep)
    print(f"HTTP Status                : {status}")
    print(f"Extracted claim            : {data.get('claim')}")
    print(f"Total evidence items       : {len(ev)}")
    print(f"Supporting (ENTAILMENT)    : {len(supporting)}")
    print(f"Contradicting (CONTRADICT) : {len(contradicting)}")
    print(f"Neutral / Skipped          : {len(neutral)}")
    print()

    print("[EVIDENCE ITEMS]")
    print(line)
    for i, e in enumerate(ev, 1):
        print(f"  [{i:02d}] NLI={e.get('nli_label'):15s} | "
              f"relevance={e.get('relevance_score', 0):.4f} | "
              f"ENT={e.get('entailment_score', 0):.3f} "
              f"CON={e.get('contradiction_score', 0):.3f} "
              f"NEU={e.get('neutral_score', 0):.3f}")
        print(f"        Source : {e.get('source_name')}")
        print(f"        URL    : {e.get('url')}")
        print(f"        Title  : {str(e.get('title', ''))[:80]}")
    print()

    print("[CREDIBILITY SCORES]")
    print(line)
    print(f"  overall_score          : {cred.get('overall_score')}")
    print(f"  support_score          : {comp.get('support_score')}")
    print(f"  contradiction_score    : {comp.get('contradiction_score')}")
    print(f"  evidence_quality_score : {comp.get('evidence_quality_score')}")
    print(f"  source_quality_score   : {comp.get('source_quality_score')}")
    print(f"  confidence             : {comp.get('confidence')}")
    print()

    print("[VERDICT]")
    print(line)
    print(f"  verdict             : {data.get('verdict')}")
    print(f"  confidence          : {data.get('confidence')}")
    print(f"  support_score       : {data.get('support_score')}")
    print(f"  contradiction_score : {data.get('contradiction_score')}")
    print()

    explanation = data.get("explanation", "")
    # Detect whether Gemini or local fallback was used
    fallback_markers = [
        "Internal rationale:",
        "This claim appears to be",
        "Insufficient reliable evidence",
        "The analysis found",
    ]
    used_gemini = not any(m in explanation for m in fallback_markers)

    print("[EXPLANATION]")
    print(line)
    print(f"  Source : {'GEMINI API (real)' if used_gemini else 'LOCAL FALLBACK (Gemini unavailable)'}")
    print(f"  Text   :")
    # Wrap explanation at 68 chars for readability
    words = explanation.split()
    line_buf = "    "
    for w in words:
        if len(line_buf) + len(w) + 1 > 70:
            print(line_buf)
            line_buf = "    " + w
        else:
            line_buf += (" " if line_buf != "    " else "") + w
    if line_buf.strip():
        print(line_buf)
    print()

    print("[COMPLETE RAW JSON]")
    print(line)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()


def main():
    for claim in CLAIMS:
        print(f"\n>>> Sending request for: {claim!r}\n")
        try:
            status, data = post_verify(claim)
            report(claim, status, data)
        except urllib.error.HTTPError as e:
            print(f"HTTP ERROR {e.code}: {e.read().decode()}")
        except Exception as exc:
            print(f"REQUEST FAILED: {exc}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
