import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EMBEDDING_DIM = 10
CONTEXT_SIZE = 2

dataset = """Enzo Ferrari was not initially interested in the idea of producing road cars when he formed Scuderia 
Ferrari in 1929, with headquarters in Modena. Scuderia Ferrari (pronounced [skudeˈriːa]) literally means "Ferrari 
Stable" and is usually used to mean "Team Ferrari." Ferrari bought,[citation needed] prepared, and fielded Alfa Romeo 
racing cars for gentleman drivers, functioning as the racing division of Alfa Romeo. In 1933, Alfa Romeo withdrew its 
in-house racing team and Scuderia Ferrari took over as its works team: the Scuderia received Alfa's Grand Prix 
cars of the latest specifications and fielded many famous drivers such as Tazio Nuvolari and Achille Varzi. In 1938, 
Alfa Romeo brought its racing operation again in-house, forming Alfa Corse in Milan and hired Enzo Ferrari as manager 
of the new racing department; therefore the Scuderia Ferrari was disbanded. In September 1939, Ferrari left Alfa 
Romeo under the provision he would not use the Ferrari name in association with races or racing cars for at least 
four years. A few days later he founded Auto Avio Costruzioni, headquartered in the facilities of the old Scuderia 
Ferrari. The new company ostensibly produced machine tools and aircraft accessories. In 1940, Ferrari produced a 
race car – the Tipo 815, based on a Fiat platform. It was the first Ferrari car and debuted at the 1940 Mille Miglia, 
but due to World War II it saw little competition. In 1943, the Ferrari factory moved to Maranello, where it has 
remained ever since. The factory was bombed by the Allies and subsequently rebuilt including works for road car 
production. The first Ferrari-badged car was the 1947 125 S, powered by a 1.5 L V12 engine; Enzo Ferrari 
reluctantly built and sold his automobiles to fund Scuderia Ferrari. The Scuderia Ferrari name was resurrected to 
denote the factory racing cars and distinguish them from those fielded by customer teams. In 1960 the company was 
restructured as a public corporation under the name SEFAC S.p.A. (Società Esercizio Fabbriche Automobili e Corse).[
15] Early in 1969, Fiat took a 50% stake in Ferrari. An immediate result was an increase in available investment 
funds, and work started at once on a factory extension intended to transfer production from Fiat's Turin plant of the 
Ferrari engined Fiat Dino. New model investment further up in the Ferrari range also received a boost. In 1988, 
Enzo Ferrari oversaw the launch of the Ferrari F40, the last new Ferrari launched before his death later that year. 
In 1989, the company was renamed Ferrari S.p.A. From 2002 to 2004, Ferrari produced the Enzo, their fastest model 
at the time, which was introduced and named in honor of the company's founder, Enzo Ferrari. It was to be called the 
F60, continuing on from the F40 and F50, but Ferrari was so pleased with it, they called it the Enzo instead. It was 
initially offered to loyal and recurring customers, each of the 399 made (minus the 400th which was donated to the 
Vatican for charity) had a price tag of $650,000 apiece (equivalent to £400,900). On 15 September 2012, 964 Ferrari 
cars worth over $162 million (£99.95 million) attended the Ferrari Driving Days event at Silverstone Circuit and 
paraded round the Silverstone Circuit setting a world record. Ferrari's former CEO and Chairman, 
Luca di Montezemolo, resigned from the company after 23 years, who was succeeded by Amedeo Felisa and finally on 3 
May 2016 Amedeo resigned and was succeeded by Sergio Marchionne, CEO and Chairman of Fiat Chrysler Automobiles, 
Ferrari's parent company. In July 2018, Marchionne was replaced by board member Louis Camilleri as CEO and by 
John Elkann as chairman. On 29 October 2014, the FCA group, resulting from the merger between manufacturers Fiat 
and Chrysler, announced the split of its luxury brand, Ferrari. The aim is to turn Ferrari into an independent brand 
which 10% of stake will be sold in an IPO in 2015. Ferrari officially priced its initial public offering at $52 a 
share after the market close on 20 October 2015. """.split()

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

        loss_history.append(epoch_loss / len(trigrams))
    print(loss_history)
