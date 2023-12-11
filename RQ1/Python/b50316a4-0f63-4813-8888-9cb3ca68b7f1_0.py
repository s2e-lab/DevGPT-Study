model = GPTLanguageModel()
model = model.to(device)  # Change here: use only 'model' instead of 'm'
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model.state_dict(), f'gpt_model_checkpoint_{iter}.pth')

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)  # Move the inputs to the same device as the model

    logits, loss = model(xb, yb)  # Change here: use 'model' instead of 'm'
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))  # Change here: use 'model' instead of 'm'
