import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ind", type=str, required=True)
parser.add_argument("--window", type=int, required=True)
parser.add_argument("--epoch", type=int, required=True)
parser.add_argument("--name", type=str, required=True)

args = parser.parse_args()
window = args.window
ind = args.ind
epoch = args.epoch
name = args.name

EMBEDDING_DIM = 10
CONTEXT_SIZE = 1 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    R = window_size
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]
    
    return list(target_words)

def get_pos_context(tokens, window):
    # we have to know the position of the token in the array
    context = []
    
    for i, token in enumerate(tokens):
        context.append([token, get_target(tokens, i, 1)])
    return context


def process_skipgrams(sentence, window, lists):
        vocab, ngrams, contexts = lists

        # change tokenizer 
        tokens = [vocab[word] for word in sentence.lower().split()]

        pos_context = get_pos_context(tokens, window)
        #[ngrams.append(i) for i in pos_context]
        #neg_context = get_neg_context(pos_context, num)

        for i, token in enumerate(pos_context):
               for j, context in enumerate(token[1]):
                        instance_context = [pos_context[i][1][j]] #+ neg_context[i][1][j]
        #                label = [1] + [0 for i in range(num)]
                        ngrams.append(token[0])
                        contexts.append(instance_context)
        #                labels.append(label)

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def train(lists, num_epochs, name):
    vocab, ngrams, contexts = lists

    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        total_loss = 0
        for instance, target in enumerate(ngrams):
            context = contexts[instance]

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor(context, dtype=torch.long)
            target = torch.tensor([target], dtype=torch.long)

            # send to device
            context_idxs = context_idxs.to(device)
            target = target.to(device)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words

            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor

            loss = loss_function(log_probs, target)

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!

    # To get the embedding of a particular word, e.g. "beauty"
    #print(model.embeddings.weight[vocab["appeal"]])
    torch.save(model, f"models\\{name}.pt")


def main(ind_data, window, epoch, name):
    df = pd.read_csv(ind_data)

    df["Text"] = df["Text"].str.lower()
    df_tokens = " ".join(df["Text"]).split()
    unique_tokens = list(set(df_tokens))
    unique_tokens.sort()

    indexes = [i for i in range(0, len(unique_tokens))]

    vocab = dict(zip(unique_tokens, indexes))
    reverse_vocab = dict(zip(indexes, unique_tokens))


    ngrams, contexts, labels = [], [], []
    df["Text"].apply(lambda x: process_skipgrams(x, window, (vocab, ngrams, contexts)))

    print(len(ngrams), len(contexts), len(labels))
    print(f"\nTarget:{ngrams[:10]}\nContexts:{contexts[:10]}\nLabels{labels[:10]}")

    train((vocab, ngrams, contexts), epoch, name)

if __name__ == "__main__":
    main(ind, window, epoch, name)