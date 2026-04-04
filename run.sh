# 1. 先建立日誌資料夾 (若尚未存在)
mkdir -p logs

# 2. 啟動三個獨立程序，各自輸出到專屬日誌檔
# CUDA_VISIBLE_DEVICES=7 nohup python main.py --pair_info /media/HDD2/hejun/LavalObjaverseDataset/experimental_pair/1_to_1_mapping_pairs.json > logs/gpu0_1to1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python main.py --pair_info /media/HDD2/hejun/LavalObjaverseDataset/experimental_pair/16_to_16_mapping_pairs.json > logs/gpu1_16to16.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python main.py --pair_info /media/HDD2/hejun/LavalObjaverseDataset/experimental_pair/64_to_64_mapping_pairs.json > logs/gpu2_64to64.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python main.py --baseline NeuralGaffer --save_gt --save_ref > logs/ng_16to16.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py --baseline NeuralGaffer --pair_info /media/HDD1/hejun/LavalObjaverseDataset/experimental_pair/16_to_16_mapping_pairs.json > logs/ng_16to16.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main.py --baseline NeuralGaffer --pair_info /media/HDD1/hejun/LavalObjaverseDataset/experimental_pair/64_to_64_mapping_pairs.json > logs/ng_64to64.log 2>&1 &
