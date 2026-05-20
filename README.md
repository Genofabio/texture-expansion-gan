# Extending Non-Stationary Texture Synthesis to Generalize on Unseen Data

![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![Task](https://img.shields.io/badge/Task-Texture%20Synthesis-green)

**Authors:** Fabio Genovese, Tommaso Querci

## Overview

This project extends the architecture proposed in *"Non-Stationary Texture Synthesis by Adversarial Expansion"* to improve texture synthesis generalization on unseen data.

Generating non-stationary textures from small image crops is a highly challenging task. To achieve coherent and realistic results, we enhanced a base GAN architecture with a VGG-19 Perceptual Loss, localized L1 loss recalibration, a robust data augmentation pipeline, and multiscale discriminators. These modifications significantly improve perceptual quality, reduce memorization behavior, and enhance structural coherence compared to the original architecture.

**Key Features:** GAN-based texture expansion, multiscale discrimination, automatic LPIPS/DISTS/NIQE/FID evaluation, and a full PyTorch implementation.

## Visual Results

### Comparison Setup: The Generalization Challenge
Evaluating generative models requires understanding their training objectives. The original baseline model was designed to perform *single-image overfitting*—memorizing thousands of crops from one specific image to expand it flawlessly. In contrast, our goal was to achieve **domain generalization** by training on a diverse dataset of 55 distinct images with heavy data augmentation.

To provide a fair and comprehensive evaluation, we compare the models in two distinct scenarios:

### Scenario A: Single-Image Expansion (Baseline's Home Ground)
Here, we test both models on the **exact image the baseline was trained on**. As expected, the baseline performs exceptionally well because it is reconstructing memorized patterns. However, our model—despite not overfitting to this specific image—successfully infers the texture rules and generates a highly coherent expansion, proving it learned the underlying structure rather than just memorizing pixels.

<table width="100%">
  <thead>
    <tr>
      <th width="20%" align="center">Ground Truth</th>
      <th width="40%" align="center">Original Model (Trained on this image)</th>
      <th width="40%" align="center">Our Model (Trained on 55 images)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/4851f37a-7462-47ec-9a71-93566d45260c" width="100%">
      </td>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/aac59df5-b08b-4449-918a-70fadc10ae53" width="100%">
      </td>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/b6735aeb-0935-4cd5-8760-4847910b8016" width="100%">
      </td>
    </tr>
  </tbody>
</table>

### Scenario B: Unseen Data Generalization (The Real Test)
This scenario evaluates true generalization by testing both networks on **completely unseen images** from our testing set. When faced with novel data, the baseline model fails to adapt, often introducing structural artifacts or unnatural tonal shifts inherited from its single training image. Our modified architecture, however, maintains strong visual, structural, and chromatic coherence across entirely new inputs.

<table width="100%">
  <thead>
    <tr>
      <th width="20%" align="center">Unseen Ground Truth</th>
      <th width="40%" align="center">Original Model (Fails to generalize)</th>
      <th width="40%" align="center">Our Model (Successful generalization)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/c69d69f4-34c1-41e9-b819-650585523d26" width="100%">
      </td>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/ce0f6690-596f-4aa0-bb66-e6adde2ca556" width="100%">
      </td>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/e7049470-b5cb-4313-8418-2f3c1f26ad65" width="100%">
      </td>
    </tr>
    <tr>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/4179a311-f5b2-4da6-8cf5-971b98bed9db" width="100%">
      </td>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/ec52e437-5ae3-4ee9-a386-04e876f9d9e7" width="100%">
      </td>
      <td align="center" valign="bottom">
        <img src="https://github.com/user-attachments/assets/8c4a5f99-dd19-4e0f-900a-0fae977b2541" width="100%">
      </td>
    </tr>
  </tbody>
</table>

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/texture-expansion-gan.git](https://github.com/yourusername/texture-expansion-gan.git)
   cd texture-expansion-gan
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision
   pip install matplotlib pyyaml tqdm lpips torchmetrics
   ```

## Project Structure

The repository is modularized to strictly separate model architecture, training pipelines, and data management.

```text
texture-expansion-gan/
├── core/
│   ├── dataset/         # Dataloaders and augmentation
│   ├── metrics/         # Cached metrics (PSNR, SSIM, LPIPS)
│   ├── model/           # Generator, Multiscale Discriminators, Losses
│   ├── pipelines/       # Isolated execution flows (train, generate, evaluate)
│   └── utils/           # Visualizations, Checkpoint I/O, File System management
├── data/                # Inputs and datasets (not tracked by git)
├── outputs/             # Generated checkpoints, logs, and evaluation metrics
├── config.yaml          # Centralized configuration (hyperparameters & routing)
└── run.py               # Main CLI entry point
```

## Usage Pipeline

All operations are managed through the central `run.py` script. The pipeline behavior is fully controlled by the `config.yaml` file.

### 1. Training
Starts the GAN training process. If previous weights are found in the checkpoint directory, training resumes automatically from the last step. If no weights are found, the environment is cleared and initialized from scratch.
```bash
python run.py train
```
Monitors, plots, and sample comparisons are generated every 100 steps inside outputs/training/logs/.

### 2. Inference (Texture Expansion)
Expands new 128x128 input textures using the trained model.
1. Place your 128x128 .png or .jpg images inside the input_folder defined in config.yaml.
2. Ensure the pre-trained .pt weights file is located in the weights_path directory.
3. Run the generator:
    ```bash
    python run.py generate
    ```

### 3. Data Preparation (Evaluation Set)
While the training dataset extracts crops dynamically *on-the-fly* to maximize variance and prevent overfitting, the evaluation dataset requires a static, immutable set of crops to ensure consistent metrics across different models.
1. Place your high-resolution test images inside the `eval_sources` directory (e.g., `data/evaluation/originals`).
2. Run the preparation script to extract the fixed crops:
    ```bash
    python run.py prepare
    ```
This will generate a static set of paired 128x128 inputs and 256x256 targets inside data/evaluation/crops.

### 4. Evaluation
Runs a quantitative and qualitative assessment over the static test dataset prepared in step 1 to measure domain generalization.
```bash
python run.py evaluate
```
This produces a final_metrics.json report containing average PSNR, SSIM, and LPIPS scores, along with visual comparison grids and metric distribution histograms saved in outputs/evaluation/.

## Methodology & Architecture

The project relies on a fully-convolutional encoder-residual-decoder Generator (featuring skip connections and a large receptive field) and a PatchGAN-based Discriminator to evaluate local realism.

To improve upon the baseline, we implemented the following key modifications:

1. **Perceptual Loss:** Introduced a VGG-19 based perceptual loss to improve semantic and structural consistency beyond pixel-level similarity.

2. **Loss Recalibration:** The original generator loss was extended to include the perceptual component:

$$
L = L_{adv} + \lambda_1 L_1 + \lambda_2 L_{style} + \lambda_{perc} L_{perc}
$$

3. **Localized L1 Loss:** Applied exclusively on the original crop region rather than the entire generated image, effectively separating reconstruction fidelity from creative texture expansion.

4. **Multiscale Discriminator:** Evaluates both local texture details and global structural coherence jointly.

5. **Data Augmentation:** A comprehensive pipeline (flips, rotations, brightness/contrast/saturation/hue shifts) to significantly boost generalization.

## Dataset

The dataset features circular wooden sections characterized by irregular concentric rings and strong non-stationary behavior. These were selected specifically to represent extremely challenging synthesis scenarios. During training, a random `128×128` crop is extracted, and the corresponding `256×256` region is used as ground truth to simulate texture expansion.

- **Configuration:** 55 source images, 400,000 random crops, 100,000 iterations (Batch Size = 8).
- **Download Data & Weights:** You can download the datasets and our best `pretrained_weights.pt` directly from the [GitHub Releases](../../releases) page of this repository.

**File Placement Guide:**
To ensure the pipelines work correctly out of the box without modifying the `config.yaml`, place the downloaded files in these exact directories (create them if they do not exist):

* **Training images** (55 files) -> `data/training/`
* **Evaluation images** (5 unseen files) -> `data/evaluation/originals/`
* **Pre-trained weights** (`.pt` file) -> `weights/`

> **Disclaimer regarding Dataset Copyright:** The dataset provided in this repository was collected and is shared exclusively for academic research and non-commercial educational purposes. The copyright of the original source images belongs to their respective owners. If you are the rights holder of any image included in the dataset and wish for it to be removed, please open an issue or contact the authors, and it will be taken down immediately.

## Limitations 

While the modifications yield substantial improvements, some limitations remain, such as:

- Tonal artifacts on unusual colors
- Memorization of strong concentric patterns
- Instability on textures far from the training distribution

## References

- Zhou et al. *"Non-Stationary Texture Synthesis by Adversarial Expansion"*, 2018
- Johnson et al. *"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"*, 2016
- Gocer, *"GAN based augmentation using a hybrid loss function for dermoscopy images"*, 2024
- Wang et al. *"Pix2PixHD"*, 2018
