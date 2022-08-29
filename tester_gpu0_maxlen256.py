# nohup python tester_gpu0_maxlen256.py > nohup_best/tester

import os
import time


# # # # albert-large-v1
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "albert-large-v1" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/albert-large-v1_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")


# # microsoft/deberta-large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "microsoft/deberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/deberta-large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
    
    
# # # # bart_large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "facebook/bart-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/bart_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
    
# robart_large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "roberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/robarta_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")


# spanbert
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "SpanBERT/spanbert-large-cased" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_5e-6/spanbert_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

    
# structbert
# bayartsogt/structbert-large
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "structbert" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_5e-6/structbert_large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

### Making best ckpt
# albert-large-v1
# for idx in range(5,7):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "albert-large-v1" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_best/albert-large-v1_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
    
# microsoft/deberta-large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "microsoft/deberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_best/deberta-large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    

# # # # bart_large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "facebook/bart-large" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_best/bart_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
# # robart_large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --epochs 10 --model_name_or_path "roberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_best/robarta_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")