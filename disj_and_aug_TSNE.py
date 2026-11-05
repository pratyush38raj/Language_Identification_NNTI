import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.manifold import TSNE


MODEL_DIR = r"C:\Users\rohit\Downloads\LOGS_AND_SAVES\disjointAug_model" # model directory containing config.json, model and preprocessor_config.json
VAL_SPEAKERS_PER_LANG = 2
SEED = 42
MAX_DURATION = 7
MAX_SAMPLES_FOR_TSNE = 1200

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

if len(valid_ds) > MAX_SAMPLES_FOR_TSNE:
    valid_ds = valid_ds.select(range(MAX_SAMPLES_FOR_TSNE))


languages = sorted(valid_ds.unique("language"))

lang_to_int = {x: i for i, x in enumerate(languages)}

speakers = sorted(valid_ds.unique("speaker_id"))

spk_to_int = {x: i for i, x in enumerate(speakers)}


embeddings = []
lang_ids = []
speaker_ids = []

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
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]          # [1, T, H]
        emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()

    embeddings.append(emb)
    lang_ids.append(lang_to_int[ex["language"]])
    speaker_ids.append(spk_to_int[ex["speaker_id"]])

embeddings = np.array(embeddings)
lang_ids = np.array(lang_ids)
speaker_ids = np.array(speaker_ids)

print("Embeddings shape:", embeddings.shape)

#tsne
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=SEED,
    init="pca",
    learning_rate="auto"
)
xy = tsne.fit_transform(embeddings)

#plotting by language
plt.figure(figsize=(10, 8))
scatter = plt.scatter(xy[:, 0], xy[:, 1], c=lang_ids, s=12, cmap="tab20")
plt.title("t-SNE of validation embeddings colored by language")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

handles, _ = scatter.legend_elements()
plt.legend(handles, languages, title="Language", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("tsne_language.png", dpi=300, bbox_inches="tight")
plt.show()

#plotting by speaker
plt.figure(figsize=(10, 8))
plt.scatter(xy[:, 0], xy[:, 1], c=speaker_ids, s=12, cmap="tab20")
plt.title("t-SNE of validation embeddings colored by speaker")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_speaker.png", dpi=300, bbox_inches="tight")
plt.show()

