# Contributing

This public repository is a cleaned release tree for review, research, and noncommercial use.

Before opening a pull request:

- Run `uv run pytest -q`.
- Run `cd frontend && bun run build && bun run test:run && bun run lint`.
- Do not add private notes, `.env` files, model checkpoints, generated media, or local machine paths.
- Keep attribution clear: the SDXL-Turbo SAEs come from Surkov et al. / EPFL `sdxl-unbox`; this project builds labels, tooling, and runtime steering on top.

By contributing, you agree that your contribution may be distributed under this repository's license and any commercial license the maintainer may separately offer for latent-dance.
