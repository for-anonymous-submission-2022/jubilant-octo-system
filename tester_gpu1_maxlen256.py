# nohup python tester_gpu1_maxlen256.py > nohup_best/tester

import os
import time


# # allenai/longformer-large-4096
# for idx in range(4,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "allenai/longformer-large-4096" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/longformer_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

# If you want to set learning rate, add "--learning_rate" (current default: 2e-5)

# google/bigbird-roberta-large
# for idx in range(7,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "google/bigbird-roberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/bigbird_roberta_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

# # robart_large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "roberta-large" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/robarta_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

# spanbert
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "SpanBERT/spanbert-large-cased" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/spanbert_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
    
# structbert
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "structbert" --batch_size 8 --max_length 256 --learning_rate 2e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_2e-6/structbert_large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    

### Making best ckpt
# longformer
# for idx in range(2,4):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "allenai/longformer-large-4096" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_best/longformer_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    

# google/bigbird-roberta-large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --epochs 10 --model_name_or_path "google/bigbird-roberta-large" --batch_size 8 --max_length 256 --learning_rate 5e-6 --train_path "PDTB/implicit_pdtb3_xval/fold_{idx}/train.tsv" --valid_path "PDTB/implicit_pdtb3_xval/fold_{idx}/dev.tsv" > nohup_best/bigbird_roberta_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    

    
