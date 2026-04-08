## INSTRUCTIONS FOR LLM AGENTS

Welcome to the XOPTPOE repository.

If you have been summoned to help build, analyze, or modify this project, please observe the following context constraints to ensure you stay aligned with the current state of the platform:

1. **The Active Codebase is `workspace_v4/`**: All active scripts, source code (`src/`), and active documentation (`docs/`) live inside `workspace_v4/`. Anything outside this directory is either legacy, data storage, or reference code.
2. **Current Project Vision**: Before proposing architectural or strategic changes, read `xoptpoe_product_brief.md` in the root directory. This contains the blueprint for evolving the current proof-of-concept into a generalized decision intelligence platform.
3. **Ignore Legacy Specs**: Do NOT use any documents inside the `archive/` directory as sources of truth. They contain definitions for v1, v2, and v3 that directly contradict current design rules. For example, any mention of an 8-sleeve universe or a locked v1 design is obsolete.
4. **Reference Implementation**: `mehmet/` contains the original Predict-Optimize-Explain research code (using a different architecture, namely pyTorch FNNs and differentiable CVXPY layers). Only reference this if you are actively debugging gradient translation between the academic paper and the v4 implementation.

Thank you.