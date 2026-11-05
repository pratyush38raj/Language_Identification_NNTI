import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


MODEL_DIR = r"C:\Users\rohit\Downloads\logs_and_saves\disjointAug_model" # # model directory containing config.json, model and preprocessor_config.json
VAL_SPEAKERS_PER_LANG = 2
SEED = 42
MAX_DURATION = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_speaker_disjoint_split(ds, val_speakers_per_lang=2, seed=42):
    rng = np.random.RandomState(seed)
    df = ds.to_pandas()
    train_parts = []
    val_parts = []

    for lang, g in df.groupby("language"):
        speakers = sorted(g["speaker_id"].dropna().unique().tolist())
        rng.shuffle(speakers)

        if len(speakers) < 2:
            g = g.sample(frac=1.0, random_state=seed)

            split_idx = max(1, int(0.8 * len(g)))
            
            g_train = g.iloc[:split_idx]
            g_val = g.iloc[split_idx:]
        else:
            n_val = min(val_speakers_per_lang, len(speakers) - 1)
            val_speakers = set(speakers[:n_val])
            g_val = g[g["speaker_id"].isin(val_speakers)]
            
            g_train = g[~g["speaker_id"].isin(val_speakers)]

        train_parts.append(g_train)
        val_parts.append(g_val)

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_out = Dataset.from_pandas(train_df, preserve_index=False)
    val_out = Dataset.from_pandas(val_df, preserve_index=False)
    return train_out, val_out

#loading model
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

#loading data
dataset = load_dataset("badrex/nnti-dataset-full")
full_ds = concatenate_datasets([dataset["train"], dataset["validation"]]).shuffle(seed=SEED)
_, valid_ds = make_speaker_disjoint_split(full_ds, val_speakers_per_lang=VAL_SPEAKERS_PER_LANG, seed=SEED)
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

#labels
id2label = model.config.id2label
label2id = model.config.label2id

def get_label_name(i):
    return id2label[i] if i in id2label else id2label[str(i)]

label_names = [get_label_name(i) for i in range(len(id2label))]

y_true = []
y_pred = []

#generating predictions
for ex in valid_ds:
    audio = ex["audio_filepath"]["array"]

    inputs = feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * MAX_DURATION),
        return_attention_mask=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()

    true_id = label2id[ex["language"]]
    y_true.append(true_id)
    y_pred.append(pred_id)

#confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - DisjointAug Updated")
plt.tight_layout()
plt.savefig("disjAUG_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

