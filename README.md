# ventiSAM Distillation

Distill Meta's Segment Anything Model (SAM) into a lightweight 5M parameter TinyViT.

```bash
git clone https://github.com/iDash3/ventiSAM-Distillation.git
cd ventiSAM-Distillation
python distill.py
```

Done. Everything is handled for you.

## Setup

Pipeline automatically:

- Installs dependencies
- Downloads dataset (interactive - choose how much) [~20 GB min free space]
- Downloads SAM weights
- Trains TinyViT student model

**Requirements:** Python 3.8+, GPU recommended

**Files:** All modules have standalone CLI interfaces and are managed through `distill.py`:

- `distill.py` - Main pipeline (run this)
- `sam_processor.py` - Dataset processor
- `train.py` - Training script
- `model.py` - TinyViT architecture

You are more than welcome to make architectural changes, train a smaller or bigger model, etc.
