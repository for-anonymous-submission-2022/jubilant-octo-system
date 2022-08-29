# nohup python tester_gpu1_full.py > nohup_full/tester

import os
import time

### Making best ckpt
# longformer
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run_full.py --epochs 10 --model_name_or_path "allenai/longformer-large-4096" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_full/longformer_large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    

# google/bigbird-roberta-large
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run_full.py --epochs 10 --model_name_or_path "google/bigbird-roberta-large" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_full/bigbird_roberta_large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")