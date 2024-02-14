import torch


def train_loop(dataloader, model, loss_fn, optimizer, alpha):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.sum(param**2)

        loss += alpha * reg_loss

        print(f"Train loss: {loss}")

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y)

    test_loss /= num_batches
    print(f"Eval loss: {test_loss:>8f}")