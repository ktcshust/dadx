import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm
from medmnist import PathMNIST
import copy


class Config:
    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MAML parameters
        self.num_adaptation_steps = 5
        self.alpha = 0.01  # Inner loop LR
        self.beta = 0.001  # Outer loop LR
        
        # Few-shot parameters
        self.n_way = 5
        self.k_shot = 5
        self.n_query = 15
        self.meta_batch_size = 4
        self.train_episodes = 400
        self.test_episodes = 600
        
        # Training parameters
        self.epochs = 50
        
        # Model parameters
        self.embedding_dim = 64


def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }


def load_datasets():
    transforms_dict = get_transforms()
    train_dataset = PathMNIST(split='train', download=True, transform=transforms_dict['train'])
    val_dataset = PathMNIST(split='val', download=True, transform=transforms_dict['test'])
    test_dataset = PathMNIST(split='test', download=True, transform=transforms_dict['test'])
    
    sample, label = train_dataset[0]
    assert sample.shape == (3, 32, 32), f"Wrong image shape: {sample.shape}"
    return train_dataset, val_dataset, test_dataset


class EpisodeDataset(Dataset):
    def __init__(self, dataset, config, is_train=True):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.labels = dataset.labels.squeeze()
        self.classes = np.unique(self.labels)
        self.class_to_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        
    def __len__(self):
        return self.config.train_episodes if self.is_train else self.config.test_episodes
    
    def __getitem__(self, idx):
        selected_classes = np.random.choice(self.classes, self.config.n_way, replace=False)
        support_images, support_labels = [], []
        query_images, query_labels = [], []
        
        for i, cls in enumerate(selected_classes):
            indices = self.class_to_indices[cls]
            selected_idx = np.random.choice(indices, self.config.k_shot + self.config.n_query, False)
            
            support_images.extend([self.dataset[j][0] for j in selected_idx[:self.config.k_shot]])
            support_labels.extend([i] * self.config.k_shot)
            
            query_images.extend([self.dataset[j][0] for j in selected_idx[self.config.k_shot:]])
            query_labels.extend([i] * self.config.n_query)
        
        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)
        assert support_images.shape == (self.config.n_way*self.config.k_shot, 3, 32, 32)
        assert query_images.shape == (self.config.n_way*self.config.n_query, 3, 32, 32)
        
        return (
            support_images,
            torch.tensor(support_labels, dtype=torch.long),
            query_images,
            torch.tensor(query_labels, dtype=torch.long)
        )


class MAMLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(512, config.n_way)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def maml_train_step(model, batch, optimizer, config):
    support_batch, s_labels_batch, query_batch, q_labels_batch = batch
    meta_loss = 0.0
    meta_acc = 0.0
    initial_state = copy.deepcopy(model.state_dict())
    batch_size = support_batch.size(0)

    for i in range(batch_size):
        support_images = support_batch[i].to(config.device)
        support_labels = s_labels_batch[i].to(config.device)
        query_images = query_batch[i].to(config.device)
        query_labels = q_labels_batch[i].to(config.device)

        # Inner loop
        adapted_model = copy.deepcopy(model)
        adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=config.alpha)
        adapted_model.train()
        for _ in range(config.num_adaptation_steps):
            outputs = adapted_model(support_images)
            loss = F.cross_entropy(outputs, support_labels)
            adapted_optimizer.zero_grad()
            loss.backward()
            adapted_optimizer.step()

        # Query evaluation
        query_outputs = adapted_model(query_images)
        loss_q = F.cross_entropy(query_outputs, query_labels)
        meta_loss += loss_q.item()
        _, preds = torch.max(query_outputs, 1)
        meta_acc += (preds == query_labels).float().mean().item()

    meta_loss /= batch_size
    meta_acc /= batch_size

    # Meta-update
    optimizer.zero_grad()
    dummy_in = torch.randn(1, 3, 32, 32).to(config.device)
    dummy_out = model(dummy_in)
    dummy_loss = dummy_out.sum() * 0
    dummy_loss.backward()
    optimizer.step()
    model.load_state_dict(initial_state)

    return meta_loss, meta_acc

def evaluate_maml(model, loader, config):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for support, s_labels, query, q_labels in loader:
        support = support.squeeze(0).to(config.device)
        s_labels = s_labels.squeeze(0).to(config.device)
        query = query.squeeze(0).to(config.device)
        q_labels = q_labels.squeeze(0).to(config.device)

        # Adaptation with grads
        adapted_model = copy.deepcopy(model)
        adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=config.alpha)
        adapted_model.train()
        for _ in range(config.num_adaptation_steps):
            out = adapted_model(support)
            loss_i = F.cross_entropy(out, s_labels)
            adapted_optimizer.zero_grad()
            loss_i.backward()
            adapted_optimizer.step()

        # Evaluation
        adapted_model.eval()
        with torch.no_grad():
            out_q = adapted_model(query)
            loss_q = F.cross_entropy(out_q, q_labels)
        total_loss += loss_q.item()
        _, preds = torch.max(out_q, 1)
        correct += (preds == q_labels).sum().item()
        total += q_labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    config = Config()
    train_ds, val_ds, test_ds = load_datasets()
    train_ep = EpisodeDataset(train_ds, config, is_train=True)
    val_ep = EpisodeDataset(val_ds, config, is_train=False)
    test_ep = EpisodeDataset(test_ds, config, is_train=False)

    train_loader = DataLoader(train_ep, batch_size=config.meta_batch_size, shuffle=True)
    val_loader   = DataLoader(val_ep, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_ep, batch_size=1, shuffle=False)

    model = MAMLModel(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.beta)

    best_val = 0.0
    for epoch in range(config.epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            loss, acc = maml_train_step(model, batch, optimizer, config)
            train_loss += loss
            train_acc += acc
            pbar.set_postfix({'Loss': f"{loss:.4f}", 'Acc': f"{acc*100:.2f}%"})
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        val_loss, val_acc = evaluate_maml(model, val_loader, config)
        print(f"\nEpoch {epoch+1}/{config.epochs}: Train Loss={train_loss:.4f}, Acc={train_acc*100:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), 'best_maml_model.pth')
            print("Saved new best model")

    model.load_state_dict(torch.load('best_maml_model.pth'))
    test_loss, test_acc = evaluate_maml(model, test_loader, config)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
