CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
  -a resnet18 \
  --lr 15.0 -j 8 \
  --batch-size 128 \
  --pretrained checkpoint_pre/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /home/zyf/AllData/ILSVRC2012-50
