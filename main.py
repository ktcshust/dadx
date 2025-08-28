import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm
from medmnist import PathMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

class Config:
    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.epochs = 80  # Tăng số epoch
        self.lr = 3e-4
        self.weight_decay = 1e-5
        
        # Few-shot parameters
        self.n_way = 5
        self.k_shot = 5
        self.n_query = 15
        self.train_episodes = 400  # Tăng số episodes huấn luyện
        self.test_episodes = 600
        
        # Model parameters
        self.embedding_dim = 256  # Tăng embedding dimension
        self.relation_dim = 64    # Tăng relation dimension

config = Config()

## 1. Data Pipeline với Augmentation mạnh hơn
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
            
            # Support set
            support_images.extend([self.dataset[j][0] for j in selected_idx[:self.config.k_shot]])
            support_labels.extend([i] * self.config.k_shot)
            
            # Query set
            query_images.extend([self.dataset[j][0] for j in selected_idx[self.config.k_shot:]])
            query_labels.extend([i] * self.config.n_query)
        
        return (
            torch.stack(support_images),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_images),
            torch.tensor(query_labels, dtype=torch.long)
        )

## 2. Model Pipeline với ResNet-18
class ResNet18Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load ResNet18 pretrained
        resnet = models.resnet18(weights=None)
        
        # Điều chỉnh cho ảnh nhỏ
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # Bỏ maxpool đầu tiên
        
        # Lấy các layer trước fc
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.projection(features)

class RelationNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.relation_net = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.relation_dim),
            nn.BatchNorm1d(config.relation_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(config.relation_dim, config.relation_dim),
            nn.BatchNorm1d(config.relation_dim),
            nn.ReLU(),
            
            nn.Linear(config.relation_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query_features, support_features, support_labels):
        n_way = len(torch.unique(support_labels))
        k_shot = support_features.size(0) // n_way
        n_query = query_features.size(0)
        
        # Tính prototype cho mỗi class
        support_features = support_features.view(n_way, k_shot, -1)
        prototypes = support_features.mean(dim=1)
        
        # Tạo các cặp query-prototype
        query_features = query_features.unsqueeze(1).expand(-1, n_way, -1)
        prototypes = prototypes.unsqueeze(0).expand(n_query, -1, -1)
        
        # Kết hợp features
        combined = torch.cat([query_features, prototypes], dim=2)
        combined = combined.view(-1, self.config.embedding_dim * 2)
        
        # Tính relation scores
        relation_scores = self.relation_net(combined)
        return relation_scores.view(n_query, n_way)

class RelationNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ResNet18Embedding(config)
        self.relation = RelationNetwork(config)
        
    def forward(self, support_images, support_labels, query_images):
        support_features = self.encoder(support_images)
        query_features = self.encoder(query_images)
        return self.relation(query_features, support_features, support_labels)

## 3. Training Pipeline
def create_dataloaders(config):
    train_dataset, val_dataset, test_dataset = load_datasets()
    
    train_episodes = EpisodeDataset(train_dataset, config, is_train=True)
    val_episodes = EpisodeDataset(val_dataset, config, is_train=False)
    test_episodes = EpisodeDataset(test_dataset, config, is_train=False)
    
    train_loader = DataLoader(train_episodes, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_episodes, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_episodes, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, optimizer, config):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training")
    for support, s_labels, query, q_labels in pbar:
        support = support.squeeze(0).to(config.device)
        s_labels = s_labels.squeeze(0).to(config.device)
        query = query.squeeze(0).to(config.device)
        q_labels = q_labels.squeeze(0).to(config.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        relation_scores = model(support, s_labels, query)
        
        # Tính loss
        loss = F.cross_entropy(relation_scores, q_labels)
        
        # Backward pass với gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Tính accuracy
        total_loss += loss.item()
        _, preds = torch.max(relation_scores, 1)
        correct += (preds == q_labels).sum().item()
        total += q_labels.size(0)
        
        pbar.set_postfix({'Loss': total_loss/(pbar.n+1), 'Acc': 100*correct/total})
    
    return total_loss/len(loader), correct/total

def evaluate(model, loader, config):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for support, s_labels, query, q_labels in pbar:
            support = support.squeeze(0).to(config.device)
            s_labels = s_labels.squeeze(0).to(config.device)
            query = query.squeeze(0).to(config.device)
            q_labels = q_labels.squeeze(0).to(config.device)
            
            relation_scores = model(support, s_labels, query)
            loss = F.cross_entropy(relation_scores, q_labels)
            
            total_loss += loss.item()
            _, preds = torch.max(relation_scores, 1)
            correct += (preds == q_labels).sum().item()
            total += q_labels.size(0)
            
            pbar.set_postfix({'Loss': total_loss/(pbar.n+1), 'Acc': 100*correct/total})
    
    return total_loss/len(loader), correct/total

## 4. Main Execution
def main():
    config = Config()
    
    # Tạo dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Khởi tạo model
    model = RelationNet(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    best_val_acc = 0
    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, config)
        val_loss, val_acc = evaluate(model, val_loader, config)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{config.epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_relationnet_resnet.pth')
            print("Saved new best model")
    
    # Đánh giá trên test set
    model.load_state_dict(torch.load('best_relationnet_resnet.pth'))
    test_loss, test_acc = evaluate(model, test_loader, config)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
