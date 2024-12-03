TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node 8 -m training.main_distill_rope \
    --dataset-type "csv" \
    --batch-size 20 \
    --train-data "/ShareGPT4V/data/share-captioner_coco_lcs_sam_1246k_1107_train_long.csv" \
    --val-data "/ShareGPT4V/data/share-captioner_coco_lcs_sam_1246k_1107_val_long.csv" \
    --logs "/logs/sharegpt4v/" \
    --warmup 1000 \
    --lr 5e-4 \
    --wd 0.1 \
    --epochs 30 \
    --workers 8 \
    --save-frequency 5 \
    --model "ViT-L-14" \
    --pretrained "openai" \
    --precision 'amp_bf16' \
    --log-every-n-steps 100 \
    --accum-freq 4 \
    --context-length 77 \
    --student-context-length 248 \
    --wandb-project-name "dense-cap-distill" \
    --loss-type "cosine" \
    --report-to "wandb" \