# %%
from datetime import datetime
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Current time: {current_time_str}")

# %%
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import torch


from datasets import load_dataset, Audio

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# %%
#checking GPU availability
print("Check if GPU available:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")


# %%
# this is a multilingual speech model, we used it as a starting point for fine-tuning
model_id = "facebook/mms-300m"

# %%
# feature extractor converts raw audio waveforms into model compatible numerical inputs
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True, # normalizing for stable learning
    return_attention_mask=True, 
)

# %%
dataset = load_dataset("badrex/nnti-dataset-full")

# %%
# inspecting the dataset
print(f"dataset['train']: {dataset['train']}")
print(f"dataset['train'][0]: {dataset['train'][0]}")

# %%
# shuffling the dataset
train_ds = dataset['train'].shuffle(seed=42)
valid_ds = dataset['validation'].shuffle(seed=42)
# resampling the dataset to 16kHz
train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

# %%
input_features_key = "input_values"  # setting the input features key

# %%

# truncating all audio clips to 7 seconds
max_duration = 7  #seconds

# %%
#label mapping
#converting language strings to numerical labels
LABELS = sorted(train_ds.unique('language'))
print(f"Languages ({len(LABELS)}): {LABELS}")

str_to_int = {label: idx for idx, label in enumerate(LABELS)}
int_to_str = {idx: label for label, idx in str_to_int.items()}
num_labels = len(LABELS)

# %%
# converting raw dataset samples into model compatible inputs using the feature extractor
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio_filepath"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )
    inputs["label"] = [str_to_int[lang] for lang in examples["language"]]
    inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]
    inputs["length"] = [len(f) for f in inputs[input_features_key]]
    return inputs

# %%
keep_cols = ['speaker_id', 'language']

train_ds_encoded = train_ds.map(
    preprocess_function,
    remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
)

valid_ds_encoded = valid_ds.map(
    preprocess_function,
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
)

# %%
config = AutoConfig.from_pretrained(model_id)
config.num_labels = num_labels
config.label2id = str_to_int
config.id2label = int_to_str

#enabling dropout for regularization
config.hidden_dropout = 0.1
config.attention_dropout = 0.1
config.activation_dropout = 0.1
config.feat_proj_dropout = 0.1

# %%
slid_model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    config=config,
)

# freezing feature extractor
slid_model.freeze_feature_extractor()

# %%
# data collator dynamically pads inputs to make all of them of same size as all samples may not be of same size
class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # preparing the batch dict in the format expected by the feature extractor
        batch = {
            input_features_key: [f[input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features]
        }
        batch = self.feature_extractor.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )
        batch["labels"] = torch.tensor(
            [f["label"] for f in features],
            dtype=torch.long
        )
        return batch

data_collator = AudioDataCollator(feature_extractor)

# %%

# setting the values for the hyperparameters
batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 30
learning_rate = 1e-5
weight_decay = 0.05

# %%
training_args = TrainingArguments(
    output_dir="./checkpoints",
    group_by_length=False,
    report_to="none",               
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    warmup_ratio=0.1,
    load_best_model_at_end=True, # loads the best model with respoct to the used metric across epochs at the end 
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,     
    push_to_hub=False,
)

# %%
# metric for evaluation
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

# %%
trainer = Trainer(
    model=slid_model,
    args=training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=4,       # stops if there is no improvement more than the threshold for 4 epochs 
            early_stopping_threshold=0.001   #threshold for improvement
        )
    ],
)

# %%
print("Train loop starting...")
trainer.train()

# %%
print("Final evaluation starting...")
metrics = trainer.evaluate()
print(metrics)

#saving the best model 
save_dir = "/home/neuronet_team159/PROJECT/baseline_improvement_model_save_final"
slid_model.save_pretrained(save_dir)
print(f"Best model saved to: {save_dir}")