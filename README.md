# RPD: Learning Efficient Crops and Weeds for Field Semantic Segmentation in Drone Images

This is the code implementation for paper "[RPD: Learning Efficient Crops and Weeds for Field Semantic Segmentation in Drone Images](https://ieeexplore.ieee.org/abstract/document/11506573)".

The paper has been accepted for publication by IEEE Transactions on Geoscience and Remote Sensing (TGRS).

Authors: Fanghui Chen; Zhen Yang; Fengyuan Ren

College of Information Science and Engineering, Lanzhou University, Lanzhou, China

This code refers to [Phenobench baseline task- semantic segmentation](https://github.com/PRBonn/phenobench-baselines/tree/main/semantic_segmentation).

If you benefit from this paper or the released code for your research, please kindly cite our work. If you have any questions about our work, feel free to contact me via e-mail (chenfh21@lzu.edu.cn).

# Reproduce & Server Guide

Detailed step-by-step instructions for reproducing on a dedicated GPU Server (SSH / RTX 4090) are provided in [SERVER_GUIDE.md](SERVER_GUIDE.md).

### Quick Start:

1. **Setup Environment**:
   ```bash
   bash setup_env.sh
   ```

2. **Train Model (Paper Config)**:
   ```bash
   # Train on GPU server
   bash run_train.sh /path/to/PhenoBench

   # Or specify custom options
   python multi_metric_train.py \
       --config ./config/config_paper.yaml \
       --dataset_dir /path/to/PhenoBench \
       --batch_size 4 \
       --max_epoch 300 \
       --num_workers 8 \
       --devices 1
   ```

3. **Summarize Results (CSV & Plots)**:
   ```bash
   python summarize_results.py ./log_dir
   ```

4. **Convert & Deploy (Re-parameterization)**:
   ```bash
   bash run_convert.sh
   ```

# Usage (Original Workflow)
* Dataset Prepare
  
  PhenoBench dataset can be downloaded [here](https://www.phenobench.org/dataset.html).
  
  CoFly dataset can be downloaded [here](https://zenodo.org/records/6697343#.YrQpwHhByV4).

- Step 1. Train the training-time model
  ```bash
  python multi_metric_train.py --config ./config/config_deeplearn.yaml --export_dir <path-to-export-directory>
  ```
 
- Step 2. Test the training-time model and convert the parallel branch in RPD blocks into a single path.
  ```bash
  python convert_multi_test.py --config ./config/config_convert.yaml --ckpt_path <path-to-ckpt> --export_dir <path-to-export-directory>
  ```
  
- Step 3. Then we need to convert weights and merge the training-time model into the inference-time model
  ```bash
  python deploy_convert_multi_test.py --config ./config/config_deploy_convert.yaml --convert_ckpt_path <path-to-ckpt> --export_dir <path-to-export-directory>
  ```

# Reference
If you find this repo useful for your research, please cite our paper:

```bibtex
@article{11506573,
  author={Chen, Fanghui and Yang, Zhen and Ren, Fengyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={RPD: Learning Efficient Crops and Weeds for Field Semantic Segmentation in Drone Images}, 
  year={2026},
  volume={64},
  pages={4408613-4408613},
  doi={10.1109/TGRS.2026.3690653}
  }
