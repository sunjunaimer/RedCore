set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 1`;
do


cmd="python3 mytest_mosei.py --dataset_mode=cmu_mosei_miss --model=mosei_ori_mmin
--log_dir=./logs/mosei --checkpoints_dir=./checkpoints/mosei --gpu_ids=$gpu
--A_type=comparE --input_dim_a=74 --norm_method=trn --embd_size_a=96 --embd_method_a=maxpool
--V_type=denseface --input_dim_v=35 --embd_size_v=96  --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=768 --embd_size_l=96 
--AE_layers=160,80,32 --n_blocks=5 --num_thread=0 --corpus=CMU_MOSEI 
--pretrained_path='checkpoints/CAP_utt_fusion_AVL_run1'
--ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0
--output_dim=3 --cls_layers=96,96 --dropout_rate=0.5
--niter=1 --niter_decay=1 --verbose --print_freq=10 --in_mem
--batch_size=128 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5         
--name=mmin_MOSEI --suffix=block_{n_blocks}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done