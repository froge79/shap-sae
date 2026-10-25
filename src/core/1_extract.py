import sys
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import last_real_token_index, load_model, load_splits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_LAYERS = tuple(range(7, 8)) # adjust this to the layers you want to extract activations from
SAE_RELEASE = "gpt2-small-resid-post-v5-32k"


def main():
    model = load_model()
    saes = load_saes()
    train_ds, val_ds, shap_ds = load_splits()
    save_split(model, saes, train_ds, "activations/probe_train")
    save_split(model, saes, val_ds, "activations/probe_val")
    save_split(model, saes, shap_ds, "activations/shap")
    print(
        f"Activations saved for layers {TARGET_LAYERS} under "
        "'activations/probe_train', 'activations/probe_val', and 'activations/shap'"
    )


def load_saes():
    saes = {}
    for layer in TARGET_LAYERS:
        sae_id = f"blocks.{layer}.hook_resid_post"
        sae = SAE.from_pretrained(SAE_RELEASE, sae_id)
        sae.to(device)
        saes[layer] = sae
    return saes


def sae_cache_key(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"


def extract_split_activations(model, saes, sentences, labels, batch_size=32):
    layers = sorted(saes.keys())
    per_layer_acts = {layer: [] for layer in layers}
    all_labels = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sents = sentences[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            tokens = model.to_tokens(batch_sents)  # [B, T]
            _, cache = model.run_with_cache_with_saes(
                tokens, saes=[saes[layer] for layer in layers]
            )

            last_idx = last_real_token_index(model, tokens)  # [B]
            batch_idx = torch.arange(tokens.shape[0], device=tokens.device)

            for layer in layers:
                sae_acts = cache[sae_cache_key(layer)]  # [B, T, F]
                sentence_acts = sae_acts[batch_idx, last_idx, :]  # [B, F]
                per_layer_acts[layer].append(sentence_acts.cpu().numpy())

            all_labels.append(np.array(batch_labels))

    labels_arr = np.concatenate(all_labels, axis=0)
    activations_by_layer = {
        layer: np.concatenate(per_layer_acts[layer], axis=0) for layer in layers
    }
    return activations_by_layer, labels_arr


def save_split(model, saes, dataset, out_dir):
    out_dir = Path(out_dir)
    acts_by_layer, labels = extract_split_activations(
        model,
        saes,
        dataset["sentence"],
        dataset["label"],
        batch_size=32,
    )
    for layer, acts in acts_by_layer.items():
        layer_dir = out_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "activations.npy", acts)
        np.save(layer_dir / "labels.npy", labels)
        print(layer_dir, acts.shape, labels.shape)





if __name__ == "__main__":
    main()
