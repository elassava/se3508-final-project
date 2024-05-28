import numpy as np
import torch
from tqdm import tqdm

#model paths
encoder_path = "saved_models/encoder.pth"
predictor_path = "saved_models/predictor.pth"

#function to train model
def model_train(trainset, trainlabels, trainlengths, encoder, predictor, optimizer, criterion, **kwargs):
    encoder.train()
    predictor.train()
    total_loss = 0
    for i in tqdm(range(0, trainset.size(1), kwargs['batch_size']), desc='Training', disable=kwargs['verbose'] < 2):
        inputs = trainset[:, i:i + kwargs['batch_size']].to(kwargs['device'])
        targets = trainlabels[i:i + kwargs['batch_size']].to(kwargs['device'])
        lengths = trainlengths[i:i + kwargs['batch_size']].cpu().long()
        
        encoder.init_hidden(inputs.size(1))
        
        optimizer.zero_grad()
        outputs = encoder(inputs, lengths)
        predictions = predictor(outputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (trainset.size(1) // kwargs['batch_size'])

#test model
def model_evaluate(validset, validlengths, encoder, predictor, **kwargs):
    encoder.eval()
    predictor.eval()
    all_predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, validset.size(1), kwargs['batch_size']), desc='Evaluating', disable=kwargs['verbose'] < 2):
            inputs = validset[:, i:i + kwargs['batch_size']].to(kwargs['device'])
            lengths = validlengths[i:i + kwargs['batch_size']].cpu().long()
            
            encoder.init_hidden(inputs.size(1))
            
            outputs = encoder(inputs, lengths)
            predictions = predictor(outputs)
            all_predictions.append(torch.argmax(predictions, dim=1).cpu().numpy())
    
    return np.concatenate(all_predictions)

def model_save(encoder, predictor):
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(predictor.state_dict(), predictor_path)
    print("Model saved successfully.")
