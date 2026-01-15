# lean-agent

RL environments for training and evaluating Lean theorem proving agents.

## Overview

This repository contains multi-turn reinforcement learning environments for Lean 4 proof formalization, built on the [verifiers](https://github.com/tensorplex/verifiers) library.

## Environments

- **minif2f_decompose** - Proof decomposition environment for miniF2F benchmark
- **lean_agent** - Multi-turn environment with subagent delegation for proof synthesis

## Setup

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

Run evaluations using the provided scripts:

```bash
# Evaluate lean_agent environment
./scripts/eval_lean_agent.sh
```

## Requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- Access to a Lean verification service
