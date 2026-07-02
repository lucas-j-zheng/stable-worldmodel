# Adversarial goal-hijacking of SOTA world models · TODOs

> Outline: `../IDEAS.md` § Goal-hijacking. Future paper (L), collaborative track —
> needs people + GPUs, not the current solo arc. *"Low-hanging-fruit paper, from
> the channel."*

**The idea.** Take a bunch of SOTA world models and find a **small adversarial
perturbation on the object** that makes the model/planner solve a *different*
goal than intended (e.g. PushT: the planner still brings the T to the target
position but at the **wrong rotation**).

**Why it's interesting:**
- **It's a bi-level optimization** — adversary vs. planner (the planner itself is
  an inner optimization, e.g. CEM/MPC). How do you attack *efficiently* through a
  planning loop rather than a single forward pass? That's the interesting
  math/algorithms question (implicit-diff through the planner, unrolled CEM,
  surrogate gradients, or zeroth-order on the adversary).
- **Why now:** reuses our existing latent-MPC harness (PushT + TwoRoom envs, CEM
  solver, frozen LeWM encoder) — the attack target is already built. "Big PR
  guaranteed," and pairs naturally with **hierarchical-JEPA exploration** (attack
  at different levels of the latent hierarchy / abstraction — see `todos/hjepa.md`).

**Research questions / eventual TODOs:**
- [ ] Minimal perturbation budget to flip the achieved goal.
- [ ] Transfer across world models / planners.
- [ ] Does planning *robustify* or *amplify* the attack vs. a reactive policy?
- [ ] Can you defend (adversarial training of the WM)?
