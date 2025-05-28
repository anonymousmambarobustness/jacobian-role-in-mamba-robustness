# SSM_robustness
This repository contains code for **"On the Role of the Jacobian in Mamba’s Adversarial
Robustness"** (NeurIPS 2025 Anonymous Submission).

We support models such as S4, DSS, S5, and Mamba, and evaluate their robustness against adversarial attacks following training or test-time adaptation (TTA), with and without Jacobian Regularization (JR).

---

## 🔧 Installation

```bash
git clone https://github.com/anonymousmambarobustness/jacobian-role-in-mamba-robustness.git
cd jacobian-role-in-mamba-robustness/SSM_robustness
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```
SSM_robustness/
│
├── checkpoints/              # Trained SSMs checkpoints
├── datasets/                 # Tiny-ImageNet and CIFAR-10
├── memo_AA_logs/             # Logs for AutoAttack (AA) runs during MEMO
├── memo_final_results/       # Destination of text files containing models performance results after MEMO TTA with/without JR
├── models/                   # Definitions of SSM architectures (S4, DSS, S5, Mamba)
├── utils/                    # Common utilities
│
├── attack_ssm.py             # PGD-10 and AA evaluation
├── download_tinyimagenet.py  # script for downloading Tiny-ImageNet dataset
├── memo_test_adapt.py        # script for applying TTA with/without JR
│
├── train_trades_cifar10.py       # CIFAR-10 training (Nat/TRADES/Madry) 
├── train_trades_tinyimagenet.py  # Tiny-ImageNet training (Nat/TRADES/Madry)
├── train_freeat_cifar10.py       # CIFAR-10 training with FreeAT
├── train_freeat_tinyimagenet.py  # Tiny-ImageNet training with FreeAT
```

---

## 🏋️‍♂️ Training with JR

You can choose between **Standard Training (Nat)** and **Adversarial Training (TRADE,Madry,FreeAT)** with:

- `--AT_type Nat` for standard training  
- `--AT_type TRADE`, `Madry`, or `FreeAT` for adversarial training

To enable or disable **JR**:

- `--reg_type gp_correct_class` enables JR 
- `--reg_type none` disables JR

To control the **strength of JR**, use the `--alpha` flag, which corresponds to λ in our paper.

### Example:

```bash
python train_trades_cifar10.py --model_name S6 --num_layers 4 --num-classes 10 --model-dir checkpoints/S6/CIFAR10/180_epochs/gp_correct_class/Nat/lr_0.001_alpha_0.001 --AT_type Nat --reg_type gp_correct_class --alpha 1e-3 --lr 1e-3 --batch-size 256 --epsilon 0.031 --step-size 0.007 --beta 6
python train_trades_cifar10.py --model_name S6 --num_layers 4 --num-classes 10 --model-dir checkpoints/S6/CIFAR10/180_epochs/none/Nat/lr_0.001_alpha_0 --AT_type Nat --reg_type none --alpha 0 --lr 1e-3 --batch-size 256 --epsilon 0.031 --step-size 0.007 --beta 6
```

---

## 🧪 Evaluation

Run PGD-10 and AutoAttack on trained models:

```bash
python attack_ssm.py --dataset CIFAR10 --num-classes 10 --model_name S6 --num_layers 4 --n_ex 10000 --AT_type Nat --lr 0.001 --epochs 180 --reg_type gp_correct_class --AA_bs 200 --batch_size 200 --test_batch_size 200 --epsilon 0.031 --step_size 0.007 --thresh 0 --num_of_seeds 1
```

---

## 🧠 Test-Time JR

We support TTA via the MEMO method, with optional JR:

- `--reg_type gp_max_output` enables JR
- `--reg_type none` disables JR

Example:

```bash
python memo_test_adapt.py --dataroot datasets/CIFAR10 --dataset CIFAR10 --num_samples 500 --num_classes 10 --model_name S6 --AT_type Nat --reg_type gp_max_output --alpha 0.1 --lr 0.005 --weight_decay 0 --niter 10 --batch_size 32 --seed 1 --calc_AA
```

---


# VMamba_robustness

This directory contains code and pretrained checkpoints for evaluating the **VMamba-T** model as part of our NeurIPS 2025 submission:  
**"On the Role of the Jacobian in Mamba’s Adversarial Robustness"** (Anonymous Submission).

We evaluate VMamba on ImageNet using PGD-10 and AA, and on ImageNet-A, ImageNet-R, and ImageNet-S using standard accuracy evaluation. All evaluations apply TTA via the MEMO method, with and without JR.

---

## 💾 Pretrained Weights

Download the VMamba-T pretrained checkpoint for ImageNet (vssm_tiny_0230_ckpt_epoch_262.pth) and place it under VMamba_robustness/pretrained_weights/ :

📎 [Google Drive – VMamba Weights](https://drive.google.com/drive/folders/1ceS0C1MGdOZcBNBLw4gESswarz4L54vD)

---

## 📁 Directory Structure

```
VMamba_robustness/
│
├── pretrained_weights/         # ImageNet-pretrained VMamba-T weights
├── imagenet_results/           # Evaluation outputs
│
├── memo_validate.py            # MEMO test-time adaptation on ImageNet
├── vmamba_robustness_inference.py  # Evaluation on ImageNet-A/ImageNet-R/ImageNet-S
├── utils/                      
```

---

## 🧠 Test-Time JR on ImageNet

**Without JR:**

```bash
python3 memo_validate.py <path_to_ImageNet_dataset> \
--num_classes 1000 --log-freq 1 --model vssm_tiny_v2 \
--checkpoint pretrained_weights/vssm_tiny_0230_ckpt_epoch_262.pth \
--normalize-model --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
--attack pgd --attack-eps 8 --attack-steps 10 --num-examples 1000 \
--lr 0.005 --num_augs 32 --niter 2 --reg_type none \
--pretraining_type Nat --alpha 0 --seed 1
```

**With JR:**

```bash
python3 memo_validate.py <path_to_ImageNet_dataset> \
--num_classes 1000 --log-freq 1 --model vssm_tiny_v2 \
--checkpoint pretrained_weights/vssm_tiny_0230_ckpt_epoch_262.pth \
--normalize-model --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
--attack pgd --attack-eps 8 --attack-steps 10 --num-examples 1000 \
--lr 0.005 --num_augs 32 --niter 2 --reg_type gp_max_output \
--pretraining_type Nat --alpha 0.01 --seed 1
```

---

## 🧪 Robustness on ImageNet-A

**Without JR:**

```bash
python3 vmamba_robustness_inference.py --dataset imagenet-a \
--data_dir <path_to_imagenet-a_dataset> --source_model_name vssm_tiny_v2 \
--num_samples 2000 --lr 5e-5 --num_augs 32 --niter 5 \
--reg_type none --pretraining_type Nat --alpha 0 --seed 1
```

**With JR:**

```bash
python3 vmamba_robustness_inference.py --dataset imagenet-a \
--data_dir <path_to_imagenet-a_dataset> --source_model_name vssm_tiny_v2 \
--num_samples 2000 --lr 5e-5 --num_augs 32 --niter 5 \
--reg_type gp_max_output --pretraining_type Nat --alpha 10 --seed 1
```


---
## 📄 Citation

*(Anonymous NeurIPS submission)*

---

## 💡 Future

We plan to extend this repository with a separate `VMamba_robustness/` directory containing code for the VMamba model experiments.
