import torch 

def calculate_accuracy(logits, labels, reduce=True):
    preds = torch.argmax(logits, dim=1)
    if reduce:
        return (preds == labels).float().mean()
    else:
        return  (preds == labels).float()