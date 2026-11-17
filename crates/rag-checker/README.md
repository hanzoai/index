# RAG Checker for Hanzo Index

This crate provides a diagnostic suite to evaluate RAG pipelines. It implements the methodology described in the NeurIPS '24 RAGChecker paper, adapted for the Hanzo stack.

## Usage
Integrate this crate into your evaluation harness to monitor:
- **Retrieval Misses**: When the index fails to provide relevant context.
- **Irrelevant Evidence**: When the index provides noise.
- **Unsupported Claims**: When the LLM hallucinates outside the context.
- **Answer Omissions**: When the LLM fails to answer despite having the evidence.
