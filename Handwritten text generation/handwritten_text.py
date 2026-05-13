import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# loading dataset

print("\nLoading dataset...")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

text = "\n".join(dataset["train"]["text"])

# taking smaller text for faster training

text = text[:40000]

print("Dataset loaded successfully")
print("Total characters:", len(text))

# vocabulary

chars = sorted(list(set(text)))

vocab_size = len(chars)

print("Vocabulary size:", vocab_size)

# character mappings

char_to_idx = {}
idx_to_char = {}

for i, ch in enumerate(chars):

    char_to_idx[ch] = i
    idx_to_char[i] = ch

# dataset class

class TextDataset(Dataset):

    def __init__(self, text, seq_len):

        self.seq_len = seq_len

        self.data = [char_to_idx[c] for c in text]

    def __len__(self):

        return len(self.data) - self.seq_len

    def __getitem__(self, index):

        x = self.data[index:index+self.seq_len]

        y = self.data[index+self.seq_len]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )

# sequence length

seq_len = 30

dataset = TextDataset(text, seq_len)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# model

class CharRNN(nn.Module):

    def __init__(self, vocab_size):

        super(CharRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 64)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.gru(x)

        output = self.fc(output[:, -1, :])

        return output

# model object

model = CharRNN(vocab_size).to(device)

print("\nModel created successfully")

# loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)

# training

epochs = 6

print("\nTraining started...\n")

for epoch in range(epochs):

    total_loss = 0

    model.train()

    for x, y in tqdm(loader):

        x = x.to(device)

        y = y.to(device)

        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("\nEpoch:", epoch + 1)

    print("Loss:", total_loss)

# saving model

torch.save(model.state_dict(), "text_model.pth")

print("\nModel saved successfully")

# text generation function

def generate_text(model, start_text, length=150):

    model.eval()

    generated_text = start_text

    input_seq = [
        char_to_idx[c]
        for c in start_text
        if c in char_to_idx
    ]

    for _ in range(length):

        current_seq = input_seq[-seq_len:]

        x = torch.tensor(
            [current_seq],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():

            output = model(x)

        # temperature for better text

        temperature = 0.7

        output = output / temperature

        probs = torch.softmax(
            output,
            dim=1
        ).cpu().numpy().ravel()

        next_index = np.random.choice(
            len(probs),
            p=probs
        )

        next_char = idx_to_char[next_index]

        generated_text += next_char

        input_seq.append(next_index)

    return generated_text

# user input loop

while True:

    user_input = input("\nEnter text: ")

    if user_input.lower() == "exit":

        print("Program ended")
        break

    generated = generate_text(
        model,
        start_text=user_input,
        length=150
    )

    print("\nGenerated text:\n")

    print(generated)

    # saving output

    with open(
        "generated_text.txt",
        "a",
        encoding="utf-8"
    ) as file:

        file.write(generated)
        file.write("\n\n")