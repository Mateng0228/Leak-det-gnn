# python -u data_gen/normal_generation.py \
#   --config configs/sim_LTA.yaml \
#   --output_root data/simulate/L-TOWN-A/normal/dataset_v1 \
#   --n_windows 200

python -u data_gen/leak_generation.py \
  --config configs/sim_LTA.yaml \
  --output_root data/simulate/L-TOWN-A/leakage/test \
  --seed 198 \
  --pipe_sample_ratio 0.5 --abrupt_per_pipe 1 --incipient_per_pipe 0 \
  --no_leak_ratio 0


# python -u models/train_predictor.py \
#   --normal_root data/simulate/L-TOWN-A/normal/dataset_v1 \
#   --out_dir results/L-TOWN-A \
#   --arch tcn \
#   --epochs 8 \
#   --steps_per_epoch 100000 \
#   --val_steps 8000 \
#   --test_steps 8000

# python -u -m models.train_detector \
#   --leak_root data/simulate/L-TOWN-A/leakage/dataset_v1 \
#   --inp_path data/raw/L-TOWN-A/L-TOWN_AreaA.inp \
#   --predictor_ckpt results/L-TOWN-A/predictor_best.ckpt \
#   --out_dir results/L-TOWN-A \
#   --epochs 8 \
#   --steps_per_epoch 100000 \
#   --batch_size 256 \
#   --log_every 1000 \
#   --topk 5

# python -u eval/event_evaluator.py \
#   --dataset_root data/simulate/L-TOWN-A/leakage/test_v1 \
#   --inp_path data/raw/L-TOWN-A/L-TOWN_AreaA.inp \
#   --predictor_ckpt results/L-TOWN-A/predictor_best.ckpt \
#   --detector_ckpt results/L-TOWN-A/detector_best.ckpt \
#   --l_pred_hours 3 --l_det_hours 3 --step_minutes 5 \
#   --include_noleak --max_leak_scens -1 --max_noleak_scens -1

# #   --out_dir data/temp \