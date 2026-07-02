# Encoder / LeWM improvements · TODOs

> Outline: `../IDEAS.md` § Encoder / LeWM.

- [ ] ◐ **Fix the early-training loss blow-up** (M). Val loss explodes ~10³ around
      epoch 4–20 then recovers (~35 wasted epochs); the 5% warmup didn't help.
      Investigate SIGReg early dynamics and the BatchNorm projector — try LayerNorm
      projector, stronger grad clipping, or a SIGReg-weight ramp-in.
- [ ] ○ **Encoder-quality → planning ablation** (S). Does a worse encoder
      (epoch-100) change downstream success? Quantifies how much planning quality
      flows from encoder quality. Only meaningful once a diffusion setup actually
      works.
