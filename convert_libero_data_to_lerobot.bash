# source examples/libero/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

export HF_LEROBOT_HOME="/mnt/sda/lwh/openpi/.cache"
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /mnt/sda/datasets/modified_libero_rlds