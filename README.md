# Why Mixture-of-Experts (MoE) Models Need Grouped Matmul (`gmm` / `ragged_dot`)

**Live Interactive Visualization:** [https://yralishe.github.io/moe-grouped-matmul/](https://yralishe.github.io/moe-grouped-matmul/)

An interactive HTML/CSS/SVG explainer visualizing why Mixture-of-Experts (MoE) architectures (such as **Qwen 3 MoE 30B–235B** and **Gemma 4 MoE**) require **Grouped Matrix Multiplication (`gmm` / `ragged_dot`)** Pallas kernels on TPUs rather than naive per-expert loops or zero-padded batched matmuls:

1. **Dense FFN Baseline (`Llama 3.2 1B`)** — Uniform `[Tokens, Hidden] × [Hidden, Intermediate]` matrix multiplication saturates `128×128` systolic MXUs.
2. **MoE Naive Approach #1 (For-Loop Over Experts)** — Launches `N` separate tiny matmuls, creating dispatch stalls and under-filled MXU tiles.
3. **MoE Naive Approach #2 (Padding All Experts to Max Size)** — Allocates `[Num_Experts, Max_Tokens, K]`, wasting massive FLOPs and HBM bandwidth multiplying padded zero rows.
4. **Grouped Matmul (`gmm` / `ragged_dot`) Solution** — Packs all routed tokens contiguously into `[Total_Tokens, K]` alongside stacked expert weights `[Num_Experts, K, N]` and `group_sizes`, executing all ragged matmuls in a single fused kernel launch with zero HBM padding waste.
