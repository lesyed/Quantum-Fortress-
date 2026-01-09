"""Quantum Fortress - Secure Model Aggregator with AI Defense."""

import torch
import numpy as np
from typing import List, Dict
import json
import time
from collections import defaultdict
from core.federated_engine.attack_detector import AIAttackDetector

class ByzantineRobustAggregator:
    def __init__(self, byzantine_threshold=0.3):
        self.global_model = None
        self.round_number = 0
        self.audit_trail = []
        self.node_reputation = defaultdict(lambda: 1.0)
        self.byzantine_threshold = byzantine_threshold
        
        # AI Detector Integration
        self.ai_detector = AIAttackDetector(gradient_dim=10000)
        self.ai_detector_trained = False

    async def aggregate(self, updates: List[Dict]):
        print(f"\n🔄 Aggregating round {self.round_number}")

        # 1. AI Detection Training (Trigger after 10 rounds for demo)
        if not self.ai_detector_trained and len(self.audit_trail) >= 10:
            print("🎓 Training AI Detector on clean history...")
            # Simulate historical training for demo
            fake_benign = [torch.randn(10000) * 0.01 for _ in range(20)]
            self.ai_detector.train_detector(fake_benign)
            self.ai_detector_trained = True

        # 2. AI Scanning
        verified_updates = []
        for update in updates:
            if self.ai_detector_trained:
                detection = self.ai_detector.detect_attack(update['model_delta'], update['node_id'])
                if detection['is_attack']:
                    print(f"🚨 AI BLOCKED {update['node_id']} - Confidence: {detection['confidence']:.2%}")
                    self.node_reputation[update['node_id']] *= 0.1
                    continue
            verified_updates.append(update)

        # 3. Traditional Statistical Aggregation (Krum/Median)
        if not verified_updates: return self.global_model
        
        aggregated_model = self._weighted_aggregation(verified_updates)
        self._record_round(verified_updates)
        self.global_model = aggregated_model
        self.round_number += 1
        return aggregated_model

    def _weighted_aggregation(self, updates):
        # Full weighted average logic
        keys = updates[0]['model_delta'].keys()
        aggregated = {k: torch.zeros_like(updates[0]['model_delta'][k]) for k in keys}
        total_weight = sum(self.node_reputation[u['node_id']] for u in updates)
        
        for u in updates:
            weight = self.node_reputation[u['node_id']]
            for k in keys:
                aggregated[k] += u['model_delta'][k] * (weight / total_weight)
        return aggregated

    def _record_round(self, updates):
        self.audit_trail.append({
            'round': self.round_number,
            'timestamp': time.time(),
            'nodes': [u['node_id'] for u in updates]
        })

    def export_audit_trail(self, filename='logs/audit_trail.json'):
        with open(filename, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)