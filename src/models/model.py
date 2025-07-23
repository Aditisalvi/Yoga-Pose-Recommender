import torch
import torch.nn as nn
from torch.utils.data import Dataset

class YogaRecommendationNN(nn.Module):
    """Neural Network for Yoga Pose Recommendation"""

    def __init__(self, user_feature_dim, asana_feature_dim, hidden_dim=128, dropout=0.3):
        super(YogaRecommendationNN, self).__init__()
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.asana_encoder = nn.Sequential(
            nn.Linear(asana_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, user_features, asana_features):
        user_embedding = self.user_encoder(user_features)
        asana_embedding = self.asana_encoder(asana_features)
        combined = torch.cat([user_embedding, asana_embedding], dim=-1)
        score = self.interaction_layer(combined)
        return score

class YogaRecommendationDataset(Dataset):
    """PyTorch Dataset for Yoga Recommendations"""

    def __init__(self, user_features, asana_features, labels):
        self.user_features = torch.FloatTensor(user_features)
        self.asana_features = torch.FloatTensor(asana_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_features[idx], self.asana_features[idx], self.labels[idx]