import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, roc_curve, auc
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from model import YogaRecommendationNN, YogaRecommendationDataset

class YogaModelTrainer:
    """Handles training and evaluation of the Yoga Recommendation model"""

    def __init__(self, user_feature_dim, asana_feature_dim, device):
        self.device = device
        self.user_feature_dim = user_feature_dim
        self.asana_feature_dim = asana_feature_dim
        self.model = None

    def train_neural_network(self, train_user_features, train_asana_features, train_labels, val_user_features, val_asana_features, val_labels):
        """Train the neural network recommendation model"""
        print("\n🚀 Training Neural Network Model...")
        train_dataset = YogaRecommendationDataset(train_user_features, train_asana_features, train_labels)
        val_dataset = YogaRecommendationDataset(val_user_features, val_asana_features, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        self.model = YogaRecommendationNN(
            user_feature_dim=self.user_feature_dim,
            asana_feature_dim=self.asana_feature_dim,
            hidden_dim=128,
            dropout=0.3
        ).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        train_losses = []
        val_losses = []
        for epoch in range(20):
            self.model.train()
            train_loss = 0.0
            for user_feat, asana_feat, labels_batch in train_loader:
                user_feat = user_feat.to(self.device)
                asana_feat = asana_feat.to(self.device)
                labels_batch = labels_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(user_feat, asana_feat).squeeze()
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for user_feat, asana_feat, labels_batch in val_loader:
                    user_feat = user_feat.to(self.device)
                    asana_feat = asana_feat.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    outputs = self.model(user_feat, asana_feat).squeeze()
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print("   ✅ Neural network training completed")
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_model(self, test_user_features, test_asana_features, test_labels):
        """Evaluate the recommendation model using various metrics"""
        print("\n📊 Evaluating Recommendation Model...")
        if self.model is None:
            print("❌ Neural network model not available for evaluation")
            return
        self.model.eval()
        predictions = []
        n_test = min(1000, len(test_labels))
        test_user_features = test_user_features[:n_test]
        test_asana_features = test_asana_features[:n_test]
        test_labels = test_labels[:n_test]
        with torch.no_grad():
            for i in range(n_test):
                user_tensor = torch.FloatTensor(test_user_features[i:i + 1]).to(self.device)
                asana_tensor = torch.FloatTensor(test_asana_features[i:i + 1]).to(self.device)
                pred = self.model(user_tensor, asana_tensor).cpu().numpy()[0][0]
                predictions.append(pred)
        predictions = np.array(predictions)
        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = (test_labels > 0.5).astype(int)
        precision = precision_score(binary_labels, binary_predictions, average='weighted')
        recall = recall_score(binary_labels, binary_predictions, average='weighted')
        f1 = f1_score(binary_labels, binary_predictions, average='weighted')
        mse = np.mean((predictions - test_labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - test_labels))
        ndcg = ndcg_score(test_labels.reshape(1, -1), predictions.reshape(1, -1), k=5)
        print(f"📈 Model Evaluation Results:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   NDCG@5: {ndcg:.4f}")
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.scatter(test_labels, predictions, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.title('Predictions vs Actual')
        plt.grid(True)
        plt.subplot(2, 3, 2)
        plt.hist(predictions, bins=20, alpha=0.7, label='Predicted')
        plt.hist(test_labels, bins=20, alpha=0.7, label='Actual')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 3, 3)
        residuals = predictions - test_labels
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Scores')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True)
        plt.subplot(2, 3, 4)
        metrics = ['Precision', 'Recall', 'F1-Score', 'NDCG@5']
        values = [precision, recall, f1, ndcg]
        plt.bar(metrics, values)
        plt.ylabel('Score')
        plt.title('Model Metrics')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.subplot(2, 3, 5)
        plt.hist(residuals, bins=20, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True)
        plt.subplot(2, 3, 6)
        fpr, tpr, _ = roc_curve(binary_labels, predictions)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'rmse': rmse,
            'mae': mae,
            'ndcg': ndcg
        }