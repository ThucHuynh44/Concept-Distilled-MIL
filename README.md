Link checkpoints: https://drive.google.com/drive/folders/1NaHEmRs1bDeiUEKapw4pBpepU0-pQ4Nw?usp=sharing

To train the model with **Concept Distillation**, run:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --task tcga_brca \
    --subtyping \
    --model_type cate_dl \
    --k 5 \
    --exp_code BRCA_Concept_Distill_V1 \
    --drop_out 0.25 \
    --lr 2e-4 \
    --early_stopping \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --log_data \
    --embed_dim 768 \
    --data_root_dir /path/to/dataset/BRCA/ \
    --split_dir /path/to/splits/tcga_brca_cross_site_val_mode_fraction \
    --concept_path /path/to/concepts/brca_concepts_titan.pt
