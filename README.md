## Environment
- Python 3.11

## Generate a test dataset
Run the following command from the repository root to generate an **example** leakage dataset:

```bash
python data_gen/leak_generation.py \
  --config configs/sim_LTA.yaml \
  --output_root data/simulate/L-TOWN-A/leakage/example \
  --seed 198 \
  --pipe_sample_ratio 0.5 --max_pipes 2 --abrupt_per_pipe 1 --incipient_per_pipe 0 \
  --no_leak_ratio 0 
```

### Notes
- The generated files will be saved under --output_root.
- For full argument descriptions: ``python data_gen/leak_generation.py --help``.
- See the ``data_gen/`` directory for implementation details and configuration options.
