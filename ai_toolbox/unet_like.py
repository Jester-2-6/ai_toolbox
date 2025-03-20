import torch
import torch.nn as nn
import tqdm

# from dataset import 

def train_unet_like(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    parellel=True,
    path=None,
    criterion=nn.CrossEntropyLoss(),
):
    # Train the model
    learning_rate = 0.1  # LEARNING_RATE
    optimizer = SGD(model.parameters(), lr=learning_rate)

    if parellel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")

    model.to(device)

    train_losses = []
    test_losses = []
    prev_loss = torch.inf

    for epoch in range(epochs):
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("Epoch: {} Loss: {:.6f}\r".format(epoch + 1, loss.item()), end="")

            epoch_loss += loss.item()

        if epoch % STEP_LR == 0:
            learning_rate = learning_rate * LR_SCALER
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        epoch_loss /= len(train_loader.dataset)
        test_loss = test_unet_like(model, test_loader, parellel=parellel)
        train_losses.append(epoch_loss)
        test_losses.append(test_loss)
        print(
            "Epoch: {} Loss: {:.6f} Test Loss: {:.6f}".format(
                epoch + 1, epoch_loss, test_loss
            )
        )

        if path is not None and prev_loss - loss > 0.01:
            print(f"Loss decreased from {prev_loss:.4f} to {loss:.4f}. Saving model.")
            save_model(model, path)

        prev_loss = loss

    return model

def test_unet_like(model, test_loader, parellel=False):
    # Test the model
    criterion = nn.MSELoss()
    test_loss = 0

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):
            if parellel:
                data = data.to(device)
                target = target.to(device)
            else:
                data = data.to("cpu")
                target = target.to("cpu")

            output = model(data)

            test_loss += criterion(output, target).item()

    test_loss /= len(test_loader.dataset)

    torch.cuda.empty_cache()

    return test_loss
