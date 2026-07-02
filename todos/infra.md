# Infra / tooling · TODOs

> Outline: `../IDEAS.md` § Infra.

- [ ] ◐ **Hard guardrail for login-node heavy work** (S). A PreToolUse hook in
      settings.json that blocks decompress/load/train commands on the Oscar login
      node (we took a usage penalty once). Belt-and-suspenders over the CLAUDE.md
      rule.
- [ ] ◐ **h5 → lance conversion for large datasets** (S). PushT h5 is 46 GB; lance
      gives faster random-access training I/O. Convert once if encoder/diffusion
      training is I/O-bound.
- [ ] ○ **"Port to new domain" script/checklist** (S). The pipeline is
      domain-portable now (only dataset/env/encoder change). A parametrized launcher
      (cache → sweep → plan) would make each new domain one command.
- [ ] ○ **Generalize the LeWM checkpoint converter** (S).
      `convert_lewm_checkpoint.py` already remaps the transformers-version skew;
      point it at cube/reacher encoders too.
