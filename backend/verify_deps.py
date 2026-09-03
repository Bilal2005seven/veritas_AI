"""
VeritasAI V1 - Dependency Verification Script
Run from inside the activated .venv to confirm all ML dependencies are usable.
"""

import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

print(f"Python version: {sys.version}")
print()

# 1. Core imports
print("--- Import checks ---")
import torch
print(f"  torch                  {torch.__version__}")

import transformers
print(f"  transformers           {transformers.__version__}")

import sentence_transformers
print(f"  sentence-transformers  {sentence_transformers.__version__}")

import numpy
print(f"  numpy                  {numpy.__version__}")

import bs4
print(f"  beautifulsoup4         {bs4.__version__}")

import lxml
print(f"  lxml                   {lxml.__version__}")
print()

# 2. Embedding model load + encode
print("--- Embedding model verification ---")
from sentence_transformers import SentenceTransformer

print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding = model.encode(["VeritasAI is verifying a news claim."])
print(f"  Embedding shape: {embedding.shape}")

assert embedding.shape == (1, 384), f"Unexpected shape: {embedding.shape}"
print("  Shape assertion: PASSED")
print()
print("All checks passed.")
