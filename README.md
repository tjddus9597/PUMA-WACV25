## Unified Metric Learning
This is a code repository for Unified metric learning. 

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- timm
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)



## Run training, dataset-specific model with conventional methods (e.g., PA, CosFace, Triplet etc.)

```
dimension_list=(128)
method=("Triplet" "MS" "Margin" "PA" "CosFace" "ArcFace" "SoftTriple" "Hyp")
loss=("Triplet" "MSLoss" "Margin" "PA" "CosFace" "ArcFace" "SoftTriple" "SupCon")
IPC=(4 4 4 0 4 0 0 0 4)
hyp_c=(0 0 0 0 0 0 0 0 0.1)
batch_size=(180, 180, 180, 180, 180, 180, 180, 180, 180)
lr=(3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5)
dataset_list=("CUB" "Cars" "SOP" "Inshop" "NAbird" "Dogs" "Flowers" "Aircraft")

for dataset in "${dataset_list[@]}"; do
    for dimension in "${dimension_list[@]}"; do
        for ((i=0; i<${#method[@]}; i++)); do
            echo "Training ${method[$i]} with ${method[$i]} Loss"
            python -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train.py \
                    --data_path ${data_path} \
                    --warmup_epochs 0 --lr ${lr[$i]} --batch_size_per_gpu ${batch_size[$i]} \
                    --weight_decay 1e-4 --clip_grad 1.0 \
                    --loss ${loss[$i]} --eval_freq 20  \
                    --use_fp16 true --emb $dimension --epochs 101 \
                    --model vit_small_patch16_224 \
                    --dataset ${dataset} --test_dataset ${dataset} \
                    --freeze false --norm_freeze false \
                    --IPC ${IPC[$i]} --sampler Unique \
                    --hyp_c ${hyp_c[$i]} \
                    --run_name Finetuned_Specific_${dataset}_${method[$i]}_IPC${IPC[$i]}
        done
    done
done
```


## Run training, universal model with PUMA

```
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train.py \
    --data_path ${data_path} \
    --lr 1e-4 --batch_size_per_gpu 180 \
    --warmup_epochs 0 --weight_decay 1e-4 \
    --eval_freq 20 --lr_decay 20 \
    --use_fp16 true --emb 128 --epochs 100 \
    --model vit_small_patch16_224 \
    --dataset All --test_dataset All \
    --freeze true --norm_freeze true \
    --loss CurricularFace --IPC 0 --alpha 32 --mrg 0.3 \
    --use_prompt true --prefix_style false --prompt_type pool \
    --component_size 20 --prompt_length 8 --prompt_lambda 1 \
    --use_adapter true --adapter_dim 128 --adapter_droppath 0.5 \
    --run_name freeze_P[0_N20L8]_A[0,11_N1D128]_CurricularFace
```

## Run Few-shot training, universal model with PUMA

```
num_shots=(1 2 4 8 16)
for ((i=0; i<${#num_shots[@]}; i++)); do
    echo "Training PUMA with ${num_shots[$i]} shot"
    python -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train.py \
        --data_path ${data_path} \
      --lr 1e-4 --batch_size_per_gpu 180 \
      --warmup_epochs 0 --weight_decay 1e-4 \
      --eval_freq $((32 / num_shots[$i])) --lr_decay cosine \
      --use_fp16 true --emb 384 --epochs $((32 / num_shots[$i] + 1)) \
      --model vit_small_patch16_224 \
      --dataset All --test_dataset All \
      --freeze true --norm_freeze true \
      --loss PA --IPC 0 --alpha 64 --mrg 0.1 \
      --use_prefix false --prompt_type pool \
      --component_size 20 --prompt_length 2 --prompt_lambda 1 \
      --use_adapter true --adapter_dim 128 --adapter_droppath 0.5 \
      --num_shot ${num_shots[$i]}  \
      --run_name freeze_P[0_N20L2]_A[0,11_N1D128]_CurricularFace_shot${num_shots[$i]}
done
```

## Setup

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm wandb timm pytorch_metric_learning
```

## Datasets

- [CUB-200](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
- [Cars-196](http://ai.stanford.edu/~jkrause/car196/car_ims.tgz) [labels](http://ai.stanford.edu/~jkrause/car196/cars_annos.mat)
- [Stanford Online Products](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
- [In-shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
    (In-shop Clothes Retrieval Benchmark -> Img -> img.zip, Eval/list_eval_partition.txt)
- [NABird](https://dl.allaboutbirds.org/nabirds)
- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs)
- [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102)
- [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft)



