import torch

def train(examples, model, epochs):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()
    loss_history = []
    #model.remap("0")
    for epoch in range(epochs):
        #random.shuffle(examples)
        epoch_loss = []
        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            prediction = model(example)
            loss = loss_function(prediction, label)
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

    return loss_history

def test(seq, t, length, model):
    dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    print(dt, t[-1] - t[-2])
    output = []
    all_t = []
    with torch.no_grad():
        for i in range(length):
            prediction = model((seq, t + dt)).reshape(1, -1, 1)
            seq = torch.cat((seq[1:], prediction), axis=0)
            all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
            t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
            output.append(prediction)

    return torch.cat(output, axis=0), torch.cat(all_t, axis=0)
