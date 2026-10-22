import datasets
from datasets import load_dataset

def load_catmus_data():
    dataset = load_dataset("CATMuS/medieval", streaming=True)
    return dataset

if True:
    dataset = load_catmus_data()
    target = 50
    sample = next(iter(dataset["train"].skip(target).take(1)))
    print("transcription:", sample["text"])