# AI Tooling

## Tools Used

- **Claude Code (Anthropic)** — primary tool for code generation and for generating the policy corpus

## How Claude Was Used

Claude Code was the main assistant throughout the development of this project. Since Python is not my primary programming language, I relied on Claude to help implement the RAG pipeline, ChromaDB integration, and Flask routes.

The general workflow was:
1. Described the project requirements to Claude in my own words to generate a step-by-step implementation plan
2. Asked Claude to generate code for each part of the plan (chunking, retrieval, prompts, guardrails, Flask routes, evaluation script)
3. Tested the application locally after each step, described any issues to Claude, and applied the fixes

I used Claude again later to get the project running after Groq removed `llama-3.1-8b-instant`, the model the app was built on.

## What Worked Well

- Breaking the project into phases (environment setup → ingestion → retrieval → web app → evaluation) made it easy to verify each step
- Claude generated accurate ChromaDB/Flask code that worked with minimal adjustment
- Describing bugs in plain language and getting targeted fixes was efficient
- The best catch was a bug in my own evaluation script: my groundedness metric compared answers against source snippets that are truncated to 200 characters, so correct answers were being scored as failures. Claude noticed the failing cases had correct answers and traced it to the metric rather than the model

## What Didn't Work Well

- Occasionally needed to re-prompt with more context when the generated code didn't match the existing file structure
- Output was often longer than asked for

## Policy Corpus Generation

The ten policy documents in `data/policies/` are synthetic — Sunshine Consulting is a fictional company and nothing in them describes a real organisation. They were generated using **Claude Code** with descriptive prompts for realistic HR policies covering PTO, remote work, information security, expenses, travel, holidays, onboarding, performance reviews, code of conduct, and acceptable use. The assignment explicitly allows the corpus to be authored "with AI assistance".
