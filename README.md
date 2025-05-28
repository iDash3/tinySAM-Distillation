# TinySAM Distillation

Distill Meta's Segment Anything Model (SAM) into a lightweight 5M parameter TinyViT.

```bash
git clone https://github.com/iDash3/tinySAM-Distillation.git
cd tinySAM-Distillation
python distill.py
```

Done. Everything is handled for you.

## Setup

**Files:** All modules can be run standalone with CLI interfaces, but managed through `distill.py`:

- `distill.py` - Main pipeline (run this)
- `sam_processor.py` - Dataset processor
- `train.py` - Training script
- `model.py` - TinyViT architecture

Pipeline automaticaly:

- Installs dependencies
- Downloads dataset (interactive - choose how much) [~20 GB min free space]
- Downloads SAM weights
- Trains TinyViT student model
