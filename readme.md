# Project: Language Identification from Audio

In this project we train a neural network model to identify the language spoken in an audio recording.
The model is built by fine-tuning the pretrained `facebook/mms-300m` multilingual speech model.

There are 2 training scripts used for implementation, one being the improvement over the provided baseline code using hyperparameter tuning and other being for task 2 where we had to overcome the problem of model learning the speaker identity either by dataset centric approach or model centric approach:

- **`baseline_improvement_2.py` (Baseline model):** It is standard fine-tuning of MMS-300M with hyperparameter tuning
- **`disjoint_and_aug_updated.py` (Task 2):** This includes speaker-disjoint validation and simple audio augmentation.

The dataset used is `badrex/nnti-dataset-full`, loaded through the HuggingFace Datasets library.

---

## Python Version

The code was developed and tested with:

- Python 3.10+

The Docker container provided in this project installs Python3 automatically.

---

## Running the Code

### `baseline_improvement_2.py` (Baseline model)

Running the baseline training script:

```bash
python baseline_improvement_2.py
```

This script:
- Loads the dataset from HuggingFace
- Preprocesses audio using the MMS feature extractor
- Fine-tunes the MMS-300M model
- Evaluates validation accuracy
- Saves the trained model

---

### Task 2 Training (`disjoint_and_aug_updated.py`)

Run the improved training script:

```bash
python disjoint_and_aug_updated.py
```

This script introduces two improvements:

#### Speaker-Disjoint Validation
- Speakers used for validation are not present in the training set.
- This prevents the model from learning speaker-specific characteristics.

#### Audio Augmentation
Two lightweight waveform augmentations are applied during training:
- Random amplitude scaling
- Small Gaussian noise

Augmentation is applied only to training samples.

---

## Model Output

After training finishes, the model is saved to a timestamped directory containing:

| File | Description |
|------|-------------|
| `pytorch_model.bin` | Trained model weights |
| `preprocessor_config.json` | Feature extractor configuration |

These files can be used later for evaluation or inference.

---

## Hardware Used

- Training was tested on a system with: **NVIDIA Tesla T4 16GB GPU**
- Mixed precision (`fp16`) training is enabled to reduce GPU memory usage.

---

## External Code and Libraries

This project uses the following open-source libraries:

| Library | Purpose |
|---------|---------|
| HuggingFace Transformers | Model architecture and fine-tuning |
| HuggingFace Datasets | Dataset loading |
| PyTorch | Deep learning framework |
| Evaluate | Metric computation |
| Scikit-learn | Additional evaluation utilities |

---

## References

- https://huggingface.co/docs/transformers
- https://huggingface.co/docs/datasets
