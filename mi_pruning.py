# mi_pruning.py
# -----------------------------------------------------------------------------
# Implements Mutual Information (MI) based layer-wise structured pruning.
#
# Based on the paper "Layer-wise Model Pruning based on Mutual Information".
# The method prunes layers in a top-down fashion, preserving neurons that
# share the most information with the preserved neurons in the layer above.
# -----------------------------------------------------------------------------

from __future__ import annotations
import copy
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# A numerically stable way to compute log determinant
def _log_det(cov: torch.Tensor) -> torch.Tensor:
    return torch.slogdet(cov).logabsdet


def _calculate_entropy(cov: torch.Tensor) -> torch.Tensor:
    """
    Calculates the entropy of a multivariate Gaussian distribution.
    H(X) = 0.5 * log((2 * pi * e)^k * det(Sigma))
    We can ignore constant terms as they cancel out in MI calculation.
    """
    if cov.dim() < 2 or cov.shape[0] != cov.shape[1]:
        # Handle scalar case (1-dim neuron)
        return 0.5 * torch.log(cov) if cov > 0 else torch.tensor(0.0)
    return 0.5 * _log_det(cov)


@torch.no_grad()
def _collect_activations(
    model: nn.Module, loader: DataLoader, device: torch.device, layer_names: List[str]
) -> Dict[str, torch.Tensor]:
    """Collects the output activations of specified layers for all samples in the loader."""
    model.eval()
    activations = {name: [] for name in layer_names}

    hooks = []
    for name in layer_names:
        layer = model.get_submodule(name)

        def hook_fn(module, inp, out, layer_name=name):
            # Handle tuple outputs from some models (e.g., BertForSequenceClassification)
            if isinstance(out, tuple):
                out_tensor = out[0]
            # Handle ModelOutput objects
            elif hasattr(out, "last_hidden_state"):
                out_tensor = out.last_hidden_state
            else:
                out_tensor = out

            # Select the [CLS] token representation if sequence data is present (e.g., [B, T, H])
            if out_tensor.dim() == 3:
                act = (
                    out_tensor[:, 0, :].detach().cpu()
                )  # Take representation of [CLS] token
            # Otherwise, it's likely a pooled or final layer output (e.g., [B, H])
            else:
                act = out_tensor.detach().cpu()

            activations[layer_name].append(act)

        hooks.append(layer.register_forward_hook(hook_fn))

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        model(**inputs)

    for hook in hooks:
        hook.remove()

    # Concatenate activations from all batches
    for name, acts in activations.items():
        if acts:
            # Concatenate along the batch dimension (dim=0). No flattening needed.
            activations[name] = torch.cat(acts, dim=0)
        else:
            activations[name] = torch.empty(0)

    return activations


def _get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Identifies prunable Linear layers in a model, typically from encoders/transformers."""
    prunable_layers = []
    # This logic can be adapted for different model architectures (e.g., ViT)
    if hasattr(model, "bert"):  # For BERT-like models
        for i, layer in enumerate(model.bert.encoder.layer):
            # Prune Attention Output and FFN layers
            prunable_layers.append(
                (
                    f"bert.encoder.layer.{i}.attention.output.dense",
                    layer.attention.output.dense,
                )
            )
            prunable_layers.append(
                (f"bert.encoder.layer.{i}.output.dense", layer.output.dense)
            )
        # Add the pooler layer
        if hasattr(model.bert, "pooler"):
            prunable_layers.append(("bert.pooler.dense", model.bert.pooler.dense))
    else:  # Fallback for other models like ViT
        for name, module in model.named_modules():
            if (
                isinstance(module, nn.Linear)
                and "classifier" not in name
                and "head" not in name
            ):
                prunable_layers.append((name, module))
    return prunable_layers


@torch.no_grad()
def mi_prune(
    model: nn.Module, loader: DataLoader, sparsity_ratio: float, device: torch.device
) -> nn.Module:
    """
    Performs MI-based pruning and returns a pruned copy of the model.
    """
    pruned_model = copy.deepcopy(model)
    prunable_layers = _get_prunable_layers(pruned_model)

    # We need layers in bottom-up order to easily access the "layer above" during the top-down loop
    prunable_layers.reverse()
    layer_names = [name for name, _ in prunable_layers]

    print("Collecting activations for MI calculation...")
    activations = _collect_activations(
        pruned_model, loader, device, layer_names + ["classifier"]
    )

    preserved_indices = {}
    # Start with the classifier layer, where all output neurons are preserved
    num_classes = pruned_model.config.num_labels
    preserved_indices["classifier"] = torch.arange(num_classes)

    # Top-down pruning loop
    # The "layer above" is the one processed in the previous iteration
    prev_layer_name = "classifier"
    for name, layer in tqdm(prunable_layers, desc="MI Pruning Layers"):
        k = int(layer.out_features * (1 - sparsity_ratio))
        if k == 0:
            continue

        # Activations of the layer above, filtered by preserved neurons
        acts_above = activations[prev_layer_name]
        if acts_above.shape[0] == 0:
            continue

        # If the layer above is not the classifier, select only the preserved neuron activations
        if prev_layer_name != "classifier":
            acts_above = acts_above[:, preserved_indices[prev_layer_name]]

        # Activations of the current layer
        acts_current = activations[name]
        if acts_current.shape[0] == 0:
            continue

        # Calculate covariance of the preserved part of the layer above
        cov_above = torch.cov(acts_above.T)
        h_above = _calculate_entropy(cov_above)

        mi_scores = []
        for i in range(
            acts_current.shape[1]
        ):  # Iterate over each neuron in the current layer
            neuron_act = acts_current[:, i].unsqueeze(1)

            # Joint activations of the current neuron and preserved neurons above
            joint_acts = torch.cat([neuron_act, acts_above], dim=1)

            # MI(neuron_i; preserved_above) = H(neuron_i) + H(preserved_above) - H(neuron_i, preserved_above)
            cov_neuron = torch.var(neuron_act.squeeze())
            cov_joint = torch.cov(joint_acts.T)

            h_neuron = _calculate_entropy(cov_neuron)
            h_joint = _calculate_entropy(cov_joint)

            mi = h_neuron + h_above - h_joint
            mi_scores.append(mi.item())

        # Select top-k neurons with the highest MI scores
        top_k_indices = torch.topk(torch.tensor(mi_scores), k=k).indices
        preserved_indices[name] = top_k_indices
        prev_layer_name = name

    # Apply masks based on preserved indices
    for name, layer in prunable_layers:
        if name in preserved_indices:
            indices = preserved_indices[name]
            mask = torch.zeros(layer.out_features, dtype=torch.bool)
            mask[indices] = True

            # Prune rows of the weight matrix and corresponding bias elements
            layer.weight.data[~mask] = 0.0
            if layer.bias is not None:
                layer.bias.data[~mask] = 0.0

    return pruned_model
