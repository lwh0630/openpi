# source examples/libero/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

export HF_LEROBOT_HOME="/mnt/data/openpi/.cache"
python examples/libero/convert_libero_data_to_lerobot.py --data_dir /mnt/data/modified_libero_rlds