# Image Generation with Flow Matching

This example demonstrates how to train a class-conditional UNet model for image generation using Flow Matching. It supports various datasets like ImageNet and CIFAR10, and also allows training on custom datasets. The implementation is based on the `guided-diffusion` repository and has been adapted for Flow Matching.

## Project Structure

The core logic is organized as follows:

-   `train.py`: The main script for training and evaluation. It handles argument parsing, dataset loading, model initialization, and the main training loop.
-   `submitit_train.py`: A wrapper script to launch training jobs on a SLURM cluster using `submitit`.
-   `models/`: Contains the model definitions.
    -   `unet.py`: The main UNet architecture, which is highly configurable for different model sizes and complexities. It includes mechanisms for timestep embedding, class conditioning, and attention.
    -   `discrete_unet.py`: A variant of the UNet for discrete flow matching.
    -   `model_configs.py`: Provides functions to instantiate models with specific configurations for different datasets (e.g., ImageNet, CIFAR10).
-   `training/`: Contains helper scripts for the training process.
    -   `train_loop.py`: The logic for a single epoch of training.
    -   `eval_loop.py`: The logic for evaluating the model, including FID score calculation.
    -   `distributed_mode.py`: Utilities for distributed training.
    -   `data_transform.py`: Defines data augmentation and preprocessing pipelines.

## Setup Instructions

1.  **Set up the Virtual Environment**: First, follow the steps in the main repository's `README.md` to create and activate the conda environment.

2.  **Install Dependencies**: Navigate to the `examples/image` directory and install the required Python packages.

    ```bash
    cd examples/image
    pip install -r requirements.txt
    ```

## Data Preparation

### ImageNet

1.  **Download and Unpack**: Download the blurred ImageNet dataset. For convenience, you can set an environment variable for the data directory.

    ```bash
    export IMAGENET_DIR=~/data/imagenet
    mkdir -p $IMAGENET_DIR
    # Assuming you have downloaded train_blurred.tar.gz to ~/Downloads
    tar -xf ~/Downloads/train_blurred.tar.gz -C $IMAGENET_DIR
    ```

2.  **Downsample Images**: Downsample the images to the desired resolution (e.g., 64x64 or 32x32). This example uses a script from a separate repository.

    ```bash
    export IMAGENET_RES=64
    cd ~/
    git clone https://github.com/PatrykChrabaszcz/Imagenet32_Scripts.git
    python Imagenet32_Scripts/image_resizer_imagent.py \
      -i ${IMAGENET_DIR}/train_blurred \
      -o ${IMAGENET_DIR}/train_blurred_${IMAGENET_RES} \
      -s $IMAGENET_RES \
      -a box \
      -r \
      -j 10
    cd - 
    ```

### Custom Dataset

You can also train the model on your own dataset. The dataset should be organized in the `ImageFolder` format:

```
my_dataset/
  train/
    class_a/
      img1.png
      img2.png
      ...
    class_b/
      img3.png
      ...
  val/
    class_a/
      ...
    class_b/
      ...
```

## Training and Evaluation

The training process is highly configurable through command-line arguments. You can find all available options in `train_arg_parser.py`.

### Local Test Run

Before launching a full training run, it's a good practice to perform a test run. This executes one step of training followed by one step of evaluation to ensure everything is set up correctly.

```bash
python train.py \
  --data_path=${IMAGENET_DIR}/train_blurred_${IMAGENET_RES}/box/ \
  --test_run
```

### Launching a Full Training on a SLURM Cluster

For larger-scale experiments, you can use `submitit_train.py` to launch the training on a SLURM cluster. This script handles the submission of the job.

```bash
python submitit_train.py \
  --data_path=${IMAGENET_DIR}/train_blurred_${IMAGENET_RES}/box/ \
  --nodes=8 \
  --batch_size=32
```

-   **Checkpoints**: Training will periodically save checkpoints to the directory specified by `--output_dir` (defaults to `./output_dir`). Only the 10 most recent checkpoints are kept by default.
-   **Resuming Training**: To resume a stopped run, you can either pass the path to a specific checkpoint using `--resume` or use the `--auto_resume` flag to automatically load from `<output_dir>/checkpoint.pth`.

### Evaluating a Model

To evaluate a trained model, use the `--eval_only` flag. You must specify the checkpoint to load using `--resume`.

-   **Generate Snapshots**: The evaluation script will generate image samples in the `/snapshots` sub-directory.
-   **Compute FID**: Add the `--compute_fid` flag to compute the Fréchet Inception Distance (FID) against the training set. The results are printed to `log.txt` in the output directory.

```bash
python submitit_train.py \
  --data_path=${IMAGENET_DIR}/train_blurred_${IMAGENET_RES}/box/ \
  --resume=./output_dir/checkpoint-899.pth \
  --compute_fid \
  --eval_only
```

### Training with a Custom Dataset

To train with a custom dataset, set `--dataset=custom2` and provide the paths to your training and validation data.

```bash
python train.py \
  --dataset=custom2 \
  --data_path=/path/to/my_dataset/train \
  --val_data_path=/path/to/my_dataset/val \
  --epochs=200 \
  --batch_size=32
```

## Pre-trained Models and Results

Here are some sample commands to reproduce results for different datasets.

| Data                 | Model type             | Epochs | FID  | Command                                                                                                                                                                                                                                                                                                                                                   |
| -------------------- | ---------------------- | ------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR-10             | Unconditional UNet     | 1800   | 2.07 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--batch_size=64 \`<br>`--nodes=1 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--cfg_scale=0.0 \`<br>`--compute_fid \`<br>`--ode_method heun2 \`<br>`--ode_options '{"nfe": 50}' \`<br>`--use_ema \`<br>`--edm_schedule \`<br>`--skewed_timesteps` |
| ImageNet32 (Blurred) | Class conditional Unet | 900    | 1.14 | `export IMAGENET_RES=32 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| ImageNet64 (Blurred) | Class conditional Unet | 900    | 1.64 | `export IMAGENET_RES=64 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| CIFAR-10 (Discrete)  | Unconditional Unet     | 2500   | 3.58 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--nodes=1 \`<br>`--discrete_flow_matching \`<br>`--batch_size=32 \`<br>`--accum_iter=1 \`<br>`--cfg_scale=0.0 \`<br>`--use_ema \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--compute_fid \`<br>`--sym_func` |

## Acknowledgements

This example partially uses code from:

-   [guided-diffusion](https://github.com/openai/guided-diffusion)
-   [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
-   [Imagenet32_Scripts](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts)

## License

The majority of the code in this example is licensed under CC-BY-NC, however portions of the project are available under separate license terms:

-   The UNet model is under the MIT license.
-   The distributed computing and the grad scaler code is under the MIT license.

## Citations

```
@inproceedings{deng2009imagenet,
  title={Imagenet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={248--255},
  year={2009},
  organization={Ieee}
}
@article{karras2022elucidating,
  title={Elucidating the design space of diffusion-based generative models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={26565--26577},
  year={2022}
}
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18},
  pages={234--241},
  year={2015},
  organization={Springer International Publishing}
}
```
