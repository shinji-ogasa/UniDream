# UniDream

Language-conditioned world model for Minecraft, built on Dreamer 4 + MineDojo.

## Project Goal

Extend Dreamer 4's world model with language conditioning so it can follow natural language instructions in Minecraft (e.g., "find diamonds", "build a house").
Core idea borrowed from Dynalang: multi-modal tokens (image + language) are fed into the same RSSM dynamics.

---

## Architecture Decisions

### DL Framework: PyTorch
- Dreamer 4 (`nicklashansen/dreamer4`) is PyTorch (torch 2.8) — we build on top of it directly
- Dynalang is JAX; we borrow the concept but re-implement in PyTorch
- VPT IDM is also PyTorch → consistent stack

### Minecraft Environment: MineDojo (not MineRL)
- MineRL: outdated, painful Java dependencies, stagnant maintenance
- MineDojo: ships MineCLIP, clean API, rich task suite (programmatic + creative)
- JARVIS-VLA uses MineStudio (MineDojo family) → direct comparison possible

### Language Encoder: MineCLIP text encoder (Phase 1)
Three candidates evaluated:

| Option | Pros | Cons |
|--------|------|------|
| MineCLIP text encoder | Minecraft-specific, STEVE-1 proven, lightweight | Low generality |
| T5-small/base (frozen) | Dynalang's choice, general-purpose, pretrained | More params |
| SentenceTransformer (all-MiniLM) | Lightweight, easy embeddings | Zero Minecraft specificity |

**Decision: MineCLIP for Phase 1, swap to T5 in Phase 2.**

Rationale:
- Fair comparison with STEVE-1 baseline in Phase 1
- Dreamer 4's Modality abstraction makes encoder swap straightforward
- MineCLIP text embedding = 512-dim → Linear → Dreamer 4 `d_model` (768)

### Action Inference: VPT IDM
- OpenAI's Inverse Dynamics Model from pretrained VPT
- Estimates actions from frame pairs; pretrained weights available
- Used to generate pseudo-action labels from YouTube videos

---

## Repository Structure

```
unidream/
├── dreamer4/                  # Fork of nicklashansen/dreamer4 (tokenizer, dynamics, model)
├── unidream/
│   ├── language_encoder.py    # MineCLIP text encoder wrapper: str → (B,T,1,512) → (B,T,1,d_model)
│   ├── language_modality.py   # Adds Modality.LANGUAGE=8, patches Dynamics.forward
│   ├── minecraft_data.py      # MineDojo / YouTube video data loader
│   ├── vpt_idm.py             # VPT IDM wrapper for pseudo-action labeling
│   └── train_language.py      # Language-conditioned dynamics training entry point
├── configs/
│   └── minecraft.yaml
└── eval/
    └── minedojo_eval.py       # Language task evaluation on MineDojo
```

---

## Dependencies

Core additions on top of Dreamer 4's environment:

```yaml
# environment.yaml extras
- minedojo
- mineclip        # or: git+https://github.com/MineDojo/MineCLIP
- sentence-transformers  # optional, for T5/MiniLM swap in Phase 2
```

---

## Implementation Phases

### Phase 1 — Language skeleton (current)
1. Add `LANGUAGE = 8` to `dreamer4/model.py` `Modality` enum
2. Write MineCLIP text encoder wrapper (`language_encoder.py`)
   - Input: `str` (or list of strings)
   - Output: `(B, T, 1, 512)` → Linear → `(B, T, 1, d_model)`
3. Patch `Dynamics.forward` to concat language tokens into the token sequence
4. Smoke-test: language tokens flow through the world model

Goal: language token enters the world model. No data, no RL yet — just the skeleton.

### Phase 2 — Data pipeline
- VPT IDM pseudo-labeling on YouTube videos
- MineDojo programmatic task data loader

### Phase 3 — Training & evaluation
- Language-conditioned world model pretraining
- MBRL fine-tuning on MineDojo tasks
- Comparison vs. STEVE-1 / JARVIS-VLA baselines

---

## Key References

- [Dreamer 4](https://github.com/nicklashansen/dreamer4)
- [Dynalang](https://arxiv.org/abs/2308.01399)
- [MineDojo](https://minedojo.org/) / [MineCLIP](https://github.com/MineDojo/MineCLIP)
- [VPT (OpenAI)](https://github.com/openai/Video-PreTraining)
- [STEVE-1](https://arxiv.org/abs/2306.00937)
