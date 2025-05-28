# TinySAM Distillation

Distill Meta's Segment Anything Model (SAM) into a lightweight 5M parameter TinyViT.

```bash
git clone <this-repo>
cd tinysamdistill
python distill.py
```

Done. Everything is handled for you:

1. Installs dependencies
2. Downloads dataset (interactive - choose how much)
3. Downloads SAM weights
4. Trains TinyViT student model

## Requirements & Files

**Requirements:** Python 3.8+, GPU recommended, ~20GB (min) free space for dataset.

**Files:** All modules can be run standalone with CLI interfaces, but managed through `distill.py`:

- `distill.py` - Main pipeline (run this)
- `sam_processor.py` - Dataset processor
- `train.py` - Training script
- `model.py` - TinyViT architecture

The pipeline handles everything automatically - dataset download, SAM weights, preprocessing, validation, model setup, and training with resume support.

That's it.
