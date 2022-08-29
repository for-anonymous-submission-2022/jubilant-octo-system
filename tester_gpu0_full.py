# nohup python tester_gpu0_full.py > nohup_full/tester

import os
import time

### Making best ckpt
# albert-large-v1
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run_full.py --epochs 10 --model_name_or_path "albert-large-v1" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_full/albert-large-v1_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
    
# microsoft/deberta-large
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_full.py --epochs 10 --model_name_or_path "microsoft/deberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_full/deberta-large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    

# # # bart_large
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_full.py --epochs 10 --model_name_or_path "facebook/bart-large" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_full/bart_large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
    
# robart_large
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_full.py --epochs 10 --model_name_or_path "roberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_full/robarta_large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")