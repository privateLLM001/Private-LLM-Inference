import torchvision
import torch.utils.data
import transformers
import datasets

def load_dataset(name, amount=None, tokenizer_name=None):
  name:str = name.lower()

  if name == "mnist":
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(0.5, 0.5)
    ])
    train_data_raw = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data_raw = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_data = []
    for t, i in enumerate(train_data_raw):
      if amount is not None and t > amount: break
      train_data.append((i[0], i[1]))
    test_data = []
    for t, i in enumerate(test_data_raw):
      if amount is not None and t > amount: break
      test_data.append((i[0], i[1]))
    return train_data, test_data # image shape: (1, 28, 28)
  
  elif name == "cifar10-224":
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
      torchvision.transforms.Resize((224, 224))
    ])
    train_data_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_data = []
    for t, i in enumerate(train_data_raw):
      if amount is not None and t > amount: break
      train_data.append((i[0], i[1]))
    test_data = []
    for t, i in enumerate(test_data_raw):
      if amount is not None and t > amount: break
      test_data.append((i[0], i[1]))
    return train_data, test_data # image shape: (3, 224, 224)
  
  elif name == "cifar10-32":
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_data_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_data = []
    for t, i in enumerate(train_data_raw):
      if amount is not None and t > amount: break
      train_data.append((i[0], i[1]))
    test_data = []
    for t, i in enumerate(test_data_raw):
      if amount is not None and t > amount: break
      test_data.append((i[0], i[1]))
    return train_data, test_data # image shape: (3, 32, 32)
  
  elif name.startswith("glue"):
    task_name = name[5:]
    print(f"Loading {name}")
    dataset = datasets.load_dataset("glue", task_name, verification_mode=datasets.VerificationMode.NO_CHECKS, save_infos=True)
    label_list = dataset["train"].features["label"].names
    num_labels = len(label_list)
    is_regression = task_name == "stsb"
    assert(tokenizer_name is not None)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    task_to_keys = {
      "cola": ("sentence", None),
      "mnli": ("premise", "hypothesis"),
      "mrpc": ("sentence1", "sentence2"),
      "qnli": ("question", "sentence"),
      "qqp": ("question1", "question2"),
      "rte": ("sentence1", "sentence2"),
      "sst2": ("sentence", None),
      "stsb": ("sentence1", "sentence2"),
      "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[task_name]
    max_seq_length = 128
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
          (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding='max_length', max_length=max_seq_length, truncation=True)
        return result
    dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_matched" if task_name == "mnli" else "validation"]
    return train_dataset, eval_dataset, num_labels, task_name

  else:
    raise Exception("Unknown dataset")

class BatchLoader:
  
  def __init__(self, dataset, batchsize, shuffle):
    self.raw = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, drop_last=True)
    self.iterator = iter(self.raw)
  
  def get(self):
    try:
      result = next(self.iterator)
      return result
    except StopIteration:
      self.iterator = iter(self.raw)
      result = next(self.iterator)
      return result

  def __len__(self):
    return len(self.raw)
