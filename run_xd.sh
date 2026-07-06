GPU_ID=$1
SEED=$2

TRAINER=MultiModalAdapterPP
CFG=x2d
EP=15

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# training on the ImageNet database
bash scripts/xd_train.sh ${SEED} ${CFG} ${TRAINER}
# testing on other databases
for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenetv2 imagenet_sketch imagenet_a imagenet_r
do
    bash scripts/xd_test.sh ${DATASET} ${SEED} ${CFG} ${TRAINER} ${EP}
done