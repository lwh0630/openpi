export HF_ENDPOINT="https://hf-mirror.com"

MAX_RETRIES=500  # 设置最大重试次数
RETRY_DELAY=1 # 每次重试前等待的秒数

RETRY_COUNT=0
until uv run huggingface-cli download \
    --repo-type "dataset" \
    openvla/modified_libero_rlds \
    --resume-download \
    --local-dir /mnt/data/modified_libero_rlds \
    --cache-dir ./cache
do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "\nError: Download failed after $MAX_RETRIES attempts."
        exit 1
    fi
    echo "\nDownload failed. Retrying in $RETRY_DELAY seconds... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_DELAY
done

echo "\nDownload successful!"