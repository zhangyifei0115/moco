CUDA_VISIBLE_DEVICES=0,1 python main_moco.py \
  -a resnet18 --lr 0.03 --batch-size 128 \
  --moco-t 0.1 --moco-k 6528 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0  /home/zyf/AllData/ILSVRC2012-100 \
  -j 8
