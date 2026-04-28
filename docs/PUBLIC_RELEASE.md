# Public Release Checklist

Before pushing the generated public worktree to GitHub:

1. From the private repo, run `./scripts/release/sync_public_tree.sh`.
2. Verify the public worktree has no `notes/`, `.env`, `.DS_Store`, `private/`, personal remote-dev helpers, or checked-in SAE weight files.
3. Verify runtime SAE weights resolve from upstream `surokpro2/sdxl-saes`.
4. Confirm `docs/LABELING_PIPELINE.md` matches the public Hugging Face dataset state.
5. Review the README narrative, images, logos, and demo links for final public positioning.
6. Run backend and frontend checks.
7. Build the public Docker image `ghcr.io/hammamiomar/latent-dance`.
8. Push the public worktree branch to the public repo as `main`.
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
cd ../hambaJuba2ba-public
git push public public-root:main
```

Keep the private repository private and continue using
`ghcr.io/hammamiomar/hambajuba2ba` for personal GPU workflows.
