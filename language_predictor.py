import torch

#function to guess the language of user input
def determine_language(input_text, encoder, predictor, args):
    #preprocess the input text
    input_seq = torch.tensor([args['vocab'].get(char, args['vocab'][args['unk_token']]) for char in input_text], dtype=torch.long).unsqueeze(1).to(args['device'])
    input_length = torch.tensor([len(input_text)], dtype=torch.long).to('cpu') 
    batch_size = 1
    
    #forward pass through the encoder nad the predictor
    encoder.init_hidden(batch_size) 
    with torch.no_grad():
        embeddings = encoder(input_seq, input_length)

    posteriors = predictor(embeddings)
    
    predicted_language_idx = torch.argmax(posteriors, dim=1).item()
    predicted_language = [language for language, idx in args['outputs'].items() if idx == predicted_language_idx][0]
    
    return predicted_language
