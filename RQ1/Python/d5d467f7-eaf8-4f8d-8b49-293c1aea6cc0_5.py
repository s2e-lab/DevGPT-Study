for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        scores = model(data)
        loss = criterion(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
