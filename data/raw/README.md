# Synthetic Data Pipeline

This directory contains the bootstrap pipeline for generating synthetic
supplier-risk datasets. To reproduce the canonical training and weekly scoring
files, run:

```bash
cd data/raw
python demo_pipeline.py
```

The script will populate `../processed/merged_training.csv` and
`../processed/new_data_weekly.csv`, train baseline models under `../../models`,
and seed the auditing logs for the demo platform.

