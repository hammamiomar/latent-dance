# Public Release Checklist

Before pushing this branch to the public GitHub repository:

1. Verify private checkout is clean except intentional private untracked directories.
2. Verify this public worktree has no `notes/`, `.env`, `.DS_Store`, private sync helpers, or checked-in SAE weight files.
3. Verify runtime SAE weights resolve from upstream `surokpro2/sdxl-saes`.
4. Confirm `docs/LABELING_PIPELINE.md` matches the public Hugging Face dataset state.
5. Review the README narrative, images, logos, and demo links for final public positioning.
6. Run backend and frontend checks.
7. Build the public Docker image.
8. Push this branch to the public repo as `main`.
9. Follow up in the personal website: update psite blog/site links and naming from the old hambajuba2ba repository to latent-dance / computers-dance language.

Completed external cleanup:

- Hugging Face dataset README attribution was updated on 2026-04-27:
  https://huggingface.co/datasets/hammamiomar/sdxl-turbo-sae-labels/commit/0355ef0982a4c8598ce9dabab96caa64c7df1ce9
- Root `.DS_Store` was removed from the Hugging Face dataset repository in the same commit.
- Hugging Face dataset README now links upstream runtime weights:
  https://huggingface.co/datasets/hammamiomar/sdxl-turbo-sae-labels/commit/a4e2c8f6e05ec0bd74e2aca84d8388558c03d2cf
- Hugging Face dataset README now points at the public latent-dance repository:
  https://huggingface.co/datasets/hammamiomar/sdxl-turbo-sae-labels/commit/8fd9cac23ee7c4b69a11f3c595d65473ce4c0b97
- Runtime SAE weight resolution from upstream `surokpro2/sdxl-saes` was verified on 2026-04-27.

Suggested push shape:

```bash
git remote add public https://github.com/hammamiomar/latent-dance.git
git push public public-main:main
```

Keep the private repository private and continue using its personal Docker/GPU workflows there.
