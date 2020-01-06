import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EMBEDDING_DIM = 5
CONTEXT_SIZE = 2

with open("dataset", "r", encoding="utf8") as f:
    dataset = f.read().split()
print(dataset)

trigrams = [([dataset[i], dataset[i + 1]], dataset[i + 2])
            for i in range(len(dataset) - 2)]

word_one_hot = {word: i for i, word in enumerate(set(dataset))}
idx_to_words = {word_one_hot[key]: key for key in word_one_hot}

print(word_one_hot)
print(idx_to_words)
vocab = set(dataset)


class Embeddings(nn.Module):
    def __init__(self, vocab_len, embedding_len, context_len):
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_len, embedding_len)
        self.layer1 = nn.Linear(context_len * embedding_len, 256)
        self.layer2 = nn.Linear(256, vocab_len)

    def forward(self, words):
        embeddings = self.embeddings(words).view(1, -1)
        x = self.layer1(embeddings)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.log_softmax(x, dim=1)

        return x


if __name__ == "__main__":
    loss_history = []
    criterion = nn.NLLLoss()
    model = Embeddings(vocab_len=len(vocab), embedding_len=EMBEDDING_DIM, context_len=CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(500):
        epoch_loss = 0
        for i, (context, target) in enumerate(trigrams):

            context_idxs = torch.tensor([word_one_hot[word] for word in context], dtype=torch.long)
            target_idxs = torch.tensor([word_one_hot[target]], dtype=torch.long)

            optimizer.zero_grad()

            output = model(context_idxs)

            loss = criterion(output, target_idxs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("Sentence was {} {} {}".format(idx_to_words[context_idxs[0].item()],
                                                 idx_to_words[context_idxs[1].item()],
                                                 idx_to_words[target_idxs.item()]))

            print("Predicted sentence is {} {} {}".format(idx_to_words[context_idxs[0].item()], idx_to_words[context_idxs[1].item()],
                                                           idx_to_words[torch.argmax(output).item()]))

            print(f"Epoch {epoch}, Iteration {i + 1}/{len(trigrams)}, Loss is {loss.item()}")
            if epoch > 900:
                time.sleep(0.1)
        loss_history.append(epoch_loss / len(trigrams))
    print(loss_history)
