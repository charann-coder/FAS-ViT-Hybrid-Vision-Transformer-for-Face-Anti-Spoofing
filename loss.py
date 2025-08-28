import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class MultiTaskLoss(nn.Module):
    """
    Combines binary cross-entropy for real/spoof detection and cross-entropy for spoof type classification.
    """
    def __init__(self, spoof_weight=0.5, label_smoothing=0.0, class_weights=None, warmup_epochs=0):
        super().__init__()
        self.spoof_weight = spoof_weight
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.ce = get_loss('ce', class_weights=class_weights, label_smoothing=label_smoothing)
        self.ce_spoof = get_loss('ce', label_smoothing=label_smoothing)

    def forward(self, outputs, real_spoof_labels, spoof_type_labels):
        real_spoof_out, spoof_type_out = outputs

        # Loss for the primary task (real vs. spoof) is calculated on all samples.
        loss_real_spoof = self.ce(real_spoof_out, real_spoof_labels)

        # Create a mask to select only the spoof samples (where label == 1).
        spoof_mask = (real_spoof_labels == 1)
        
        loss_spoof_type = torch.tensor(0.0, device=outputs[0].device)
       
        if spoof_mask.any():
          
            filtered_spoof_out = spoof_type_out[spoof_mask]
            filtered_spoof_labels = spoof_type_labels[spoof_mask]
            
            
            loss_spoof_type = self.ce_spoof(filtered_spoof_out, filtered_spoof_labels)

        
        current_spoof_weight = self.spoof_weight
        if self.epoch < self.warmup_epochs and self.warmup_epochs > 0:
            current_spoof_weight = self.spoof_weight * (self.epoch / self.warmup_epochs)
        
        return (1.0 - current_spoof_weight) * loss_real_spoof + current_spoof_weight * loss_spoof_type

def get_loss(loss_name='ce', class_weights=None, label_smoothing=0.0):
    """
    Factory function to get a loss function.
    """
    if loss_name == 'ce':
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

def get_multitask_loss(spoof_weight=0.5, label_smoothing=0.0, class_weights=None, warmup_epochs=0):
    """
    Factory function to get the multitask loss.
    """
    return MultiTaskLoss(spoof_weight=spoof_weight, label_smoothing=label_smoothing,
                         class_weights=class_weights, warmup_epochs=warmup_epochs)