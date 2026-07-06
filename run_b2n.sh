GPU_ID=$1
SEED=$2

TRAINER=MultiModalAdapterPP
CFG=b2n_cupl
EP=50

export CUDA_VISIBLE_DEVICES=${GPU_ID}

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
    bash scripts/base2new_train.sh ${DATASET} ${SEED} ${CFG} ${TRAINER}
    bash scripts/base2new_test.sh ${DATASET} ${SEED} ${CFG} ${TRAINER} ${EP}
done
