import torch
import torch.nn as nn


def _batch_losses(model, batch, device, criterion_act, criterion_time):
    x_acts, x_times, y_acts, y_times = (tensor.to(device) for tensor in batch)
    logits_act, pred_time = model(x_acts, x_times)
    loss_act = criterion_act(logits_act, y_acts)
    loss_time = criterion_time(pred_time.squeeze(-1), y_times)
    return loss_act, loss_time, logits_act, y_acts

@torch.no_grad()
def evaluate(model, loader, device, criterion_act, criterion_time):
    model.eval()
    total_act = total_time = 0.0
    total = correct = 0
    for batch in loader:
        loss_act, loss_time, logits_act, y_acts = _batch_losses(model, batch, device, criterion_act, criterion_time)
        n = y_acts.size(0)
        total_act += loss_act.item() * n
        total_time += loss_time.item() * n
        correct += (logits_act.argmax(dim=1) == y_acts).sum().item()
        total += n
    total = max(total, 1)
    return total_act / total, total_time / total, correct / total


def train_sutran_model(model, train_loader, val_loader, device, epochs, lr, weight_decay):
    model = model.to(device)
    criterion_act = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_act = running_time = 0.0
        n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss_act, loss_time, _, y_acts = _batch_losses(model, batch, device, criterion_act, criterion_time)    
            (loss_act + loss_time).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_act += loss_act.item() * y_acts.size(0)
            running_time += loss_time.item() * y_acts.size(0)
            n += y_acts.size(0)
            
        val_act, val_time, val_acc = evaluate(model, val_loader, device, criterion_act, criterion_time)
        print(f"Epoch {epoch:02d} | Train Act Loss: {running_act / max(n, 1):.4f} | "
              f"Train Time MSE: {running_time / max(n, 1):.4f} | Val Act Loss: {val_act:.4f} | "
              f"Val Time MSE: {val_time:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return model