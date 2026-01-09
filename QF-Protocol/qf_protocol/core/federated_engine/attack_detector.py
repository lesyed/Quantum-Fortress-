"""Quantum Fortress - AI-Powered Attack Detector
Uses neural networks to detect poisoning attacks that bypass statistics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import json
from datetime import datetime

class GradientEncoder(nn.Module):
    """Autoencoder that learns what 'normal' gradients look like."""
    def __init__(self, gradient_dim=10000, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gradient_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, gradient_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
        
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error

class TrajectoryLSTM(nn.Module):
    """LSTM that learns gradient evolution patterns."""
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        attack_prob = self.classifier(last_hidden)
        return attack_prob

class PoisoningAttackGenerator:
    """Generates synthetic poisoning attacks for training."""
    @staticmethod
    def label_flipping_attack(gradients: torch.Tensor, flip_ratio=0.3):
        attacked = gradients.clone()
        num_flip = int(gradients.numel() * flip_ratio)
        indices = torch.randperm(gradients.numel())[:num_flip]
        attacked.view(-1)[indices] *= -1
        return attacked

    @staticmethod
    def backdoor_attack(gradients: torch.Tensor, trigger_magnitude=10.0):
        attacked = gradients.clone()
        attacked[:100] += torch.randn(100) * trigger_magnitude
        return attacked

    @staticmethod
    def byzantine_attack(gradients: torch.Tensor, scale=100.0):
        return gradients * scale

    @staticmethod
    def subtle_drift_attack(gradients: torch.Tensor, drift_vector: torch.Tensor, epsilon=0.1):
        return gradients + epsilon * drift_vector

class AIAttackDetector:
    """Main attack detection system."""
    def __init__(self, gradient_dim=10000):
        self.gradient_dim = gradient_dim
        self.autoencoder = GradientEncoder(gradient_dim)
        self.trajectory_lstm = TrajectoryLSTM()
        self.attack_generator = PoisoningAttackGenerator()
        self.gradient_history = deque(maxlen=50)
        self.detection_history = []
        self.reconstruction_threshold = None
        self.is_trained = False

    def train_detector(self, benign_gradients: List[torch.Tensor], epochs=20):
        print("🎓 Training AI Attack Detector...")
        train_data, train_labels = [], []
        
        for grad in benign_gradients:
            flat_grad = self._flatten_gradient(grad)
            train_data.append(flat_grad)
            train_labels.append(0)

        for grad in benign_gradients[:len(benign_gradients)//2]:
            flat_grad = self._flatten_gradient(grad)
            train_data.append(self.attack_generator.label_flipping_attack(flat_grad))
            train_labels.append(1)
            train_data.append(self.attack_generator.backdoor_attack(flat_grad))
            train_labels.append(1)

        train_data = torch.stack(train_data)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)

        self._train_autoencoder(train_data[train_labels == 0], epochs=epochs)
        self._train_trajectory_lstm(train_data, train_labels, epochs=epochs)
        self._compute_thresholds(train_data[train_labels == 0])
        self.is_trained = True
        print("✅ AI Attack Detector trained successfully!")

    def _train_autoencoder(self, benign_data, epochs):
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon, _ = self.autoencoder(benign_data)
            loss = criterion(recon, benign_data)
            loss.backward()
            optimizer.step()

    def _train_trajectory_lstm(self, data, labels, epochs):
        optimizer = torch.optim.Adam(self.trajectory_lstm.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        # Simplified trajectory training for demo
        for epoch in range(epochs):
            optimizer.zero_grad()
            # We assume a batch of latents for training
            _, latents = self.autoencoder(data)
            # Add fake sequence dimension for demo
            seq_data = latents.unsqueeze(1).repeat(1, 5, 1)
            preds = self.trajectory_lstm(seq_data)
            loss = criterion(preds, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

    def _compute_thresholds(self, benign_data):
        errors = self.autoencoder.get_reconstruction_error(benign_data)
        self.reconstruction_threshold = torch.quantile(errors, 0.95).item()

    def detect_attack(self, gradient: Dict[str, torch.Tensor], node_id: str) -> Dict:
        if not self.is_trained:
            return {'is_attack': False, 'confidence': 0.0, 'reason': 'Not trained', 'node_id': node_id}
        
        flat_grad = self._flatten_gradient(gradient)
        recon_error = self.autoencoder.get_reconstruction_error(flat_grad.unsqueeze(0)).item()
        
        # Combined logic
        is_attack = recon_error > self.reconstruction_threshold
        confidence = 0.95 if is_attack else 0.1
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'node_id': node_id,
            'is_attack': is_attack,
            'confidence': confidence,
            'reason': "Abnormal pattern" if is_attack else "Normal",
            'recon_error': recon_error
        }
        self.detection_history.append(result)
        return result

    def _flatten_gradient(self, gradient) -> torch.Tensor:
        if isinstance(gradient, dict):
            return torch.cat([v.flatten() for k, v in sorted(gradient.items())])[:self.gradient_dim]
        return gradient.flatten()[:self.gradient_dim]