"""SAE feature labeling utilities.

Utilities for reproducing, inspecting, and validating the published SAE feature
labels. Stage 1 uses optional Modal infrastructure and is not required for
normal hambajuba2ba runtime.

Available stages (each a standalone script with file I/O boundaries):
    0. clustering      — UMAP + HDBSCAN decoder weight grouping (done)
    1. stage1_generate — Image gen + SAE activation logging (optional Modal)
    2. stage2_select   — Feature ranking, image selection, spatial cropping
    3. stage3_annotate — VLM ensemble annotation via OpenRouter
    4. stage4_supplement — TF-IDF prompt analysis, spatial activation maps
    5. stage5_fuse     — Sentence embedding fusion → final labels
    6. stage6_validate — Ground-truth gate and detection scoring
    7. stage7_factors  — NMF factor grouping → user-facing concepts

Shared modules:
    config      — Paths, hyperparams, known feature ground truth
    utils       — UNet navigation, SAE weight loading, image generation
    openrouter  — Async OpenRouter API client with retries + response parsing
    modal_app   — Optional Modal deployment wrapper for Stage 1 reproduction
"""
