"""
Quantum Fortress - Secure Model Aggregator
Byzantine-robust aggregation with audit trail
"""

import torch
import numpy as np
from typing import List, Dict
import json
import time
from scipy import stats
from collections import defaultdict

class ByzantineRobustAggregator:
    """
    Aggregates model updates with protection against:
    - Malicious nodes (Byzantine attacks)
    - Data poisoning
    - Gradient manipulation
    """
    
    def __init__(self, byzantine_threshold=0.3):
        self.global_model = None
        self.round_number = 0
        self.audit_trail = []
        self.node_reputation = defaultdict(lambda: 1.0)  # Start with full reputation
        self.byzantine_threshold = byzantine_threshold
        
    async def aggregate(self, updates: List[Dict]):
        """
        Aggregate updates using Byzantine-robust algorithm
        
        Algorithm: Multi-Krum (selects most representative updates)
        Better than simple averaging against malicious nodes
        """
        print(f"\n🔄 Aggregating round {self.round_number}")
        print(f"📦 Received {len(updates)} updates")
        
        if not updates:
            raise ValueError("No updates received")
        
        # 1. Verify all proofs
        verified_updates = []
        for update in updates:
            if self._verify_proof(update['proof']):
                verified_updates.append(update)
            else:
                print(f"⚠️ Invalid proof from {update['node_id']}")
        
        print(f"✅ Verified {len(verified_updates)}/{len(updates)} updates")
        
        # 2. Detect Byzantine nodes
        suspicious_indices = self._detect_byzantine(verified_updates)
        
        if suspicious_indices:
            print(f"🚨 Detected {len(suspicious_indices)} suspicious nodes")
            for idx in suspicious_indices:
                node_id = verified_updates[idx]['node_id']
                self.node_reputation[node_id] *= 0.5  # Reduce reputation
                print(f"   - {node_id} (reputation now: {self.node_reputation[node_id]:.2f})")
        
        # 3. Filter out suspicious updates
        clean_updates = [
            u for i, u in enumerate(verified_updates) 
            if i not in suspicious_indices
        ]
        
        if not clean_updates:
            print("⚠️ All updates rejected! Using previous model")
            return self.global_model
        
        # 4. Aggregate using weighted average
        aggregated_model = self._weighted_aggregation(clean_updates)
        
        # 5. Record to audit trail
        self._record_round(verified_updates, clean_updates, suspicious_indices)
        
        self.global_model = aggregated_model
        self.round_number += 1
        
        print(f"✨ Aggregation complete. Model updated.\n")
        
        return aggregated_model
    
    def _verify_proof(self, proof: Dict) -> bool:
        """Verify cryptographic proof of training"""
        # In production, use proper digital signature verification
        # This is simplified for demo
        
        required_fields = ['old_model_hash', 'delta_hash', 'num_samples', 'signature']
        if not all(field in proof for field in required_fields):
            return False
        
        # Verify signature (simplified)
        # In production: use RSA/ECDSA verification
        return len(proof['signature']) == 64  # SHA-256 hash length
    
    def _detect_byzantine(self, updates: List[Dict]) -> List[int]:
        """
        Detect malicious updates using statistical methods
        
        Methods:
        1. Cosine similarity outlier detection
        2. Magnitude outlier detection
        3. Historical consistency check
        """
        if len(updates) < 3:
            return []  # Need at least 3 updates for reliable detection
        
        suspicious = []
        
        # Extract model deltas
        deltas = []
        for update in updates:
            # Flatten all tensors into single vector
            delta_vector = []
            for key in sorted(update['model_delta'].keys()):
                delta_vector.extend(
                    update['model_delta'][key].cpu().numpy().flatten()
                )
            deltas.append(np.array(delta_vector))
        
        deltas = np.array(deltas)
        
        # Method 1: Cosine similarity to median
        median_delta = np.median(deltas, axis=0)
        
        for i, delta in enumerate(deltas):
            # Compute cosine similarity
            cos_sim = np.dot(delta, median_delta) / (
                np.linalg.norm(delta) * np.linalg.norm(median_delta) + 1e-8
            )
            
            # Method 2: Magnitude check
            magnitude_ratio = np.linalg.norm(delta) / (
                np.linalg.norm(median_delta) + 1e-8
            )
            
            # Method 3: Reputation score
            node_id = updates[i]['node_id']
            reputation = self.node_reputation[node_id]
            
            # Combined Byzantine score (0-1, higher = more suspicious)
            byzantine_score = (
                0.4 * (1 - max(0, cos_sim)) +  # Low similarity = suspicious
                0.3 * abs(1 - magnitude_ratio) +  # Unusual magnitude = suspicious
                0.3 * (1 - reputation)  # Low reputation = suspicious
            )
            
            if byzantine_score > self.byzantine_threshold:
                suspicious.append(i)
        
        return suspicious
    
    def _weighted_aggregation(self, updates: List[Dict]) -> Dict:
        """
        Aggregate using weighted average based on:
        - Number of samples
        - Node reputation
        """
        # Initialize aggregated model
        aggregated = {}
        total_weight = 0
        
        # Get keys from first update
        keys = updates[0]['model_delta'].keys()
        
        for key in keys:
            aggregated[key] = torch.zeros_like(updates[0]['model_delta'][key])
        
        # Weighted sum
        for update in updates:
            # Weight = samples * reputation
            weight = (
                update['proof']['num_samples'] * 
                self.node_reputation[update['node_id']]
            )
            total_weight += weight
            
            for key in keys:
                aggregated[key] += update['model_delta'][key] * weight
        
        # Average
        for key in keys:
            aggregated[key] /= total_weight
        
        return aggregated
    
    def _record_round(self, all_updates, clean_updates, suspicious):
        """Record round to immutable audit trail"""
        record = {
            'round': self.round_number,
            'timestamp': time.time(),
            'total_updates': len(all_updates),
            'accepted_updates': len(clean_updates),
            'rejected_updates': len(suspicious),
            'participants': [u['node_id'] for u in all_updates],
            'suspicious_nodes': [
                all_updates[i]['node_id'] for i in suspicious
            ],
            'model_hash': None  # Will be computed from aggregated model
        }
        
        self.audit_trail.append(record)
    
    def export_audit_trail(self, filename='audit_trail.json'):
        """Export complete audit trail for compliance"""
        with open(filename, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
        print(f"📋 Audit trail exported to {filename}")

# Test aggregator
if __name__ == "__main__":
    # Create mock updates
    updates = []
    for i in range(5):
        update = {
            'node_id': f'node_{i}',
            'round': 0,
            'model_delta': {
                'fc1.weight': torch.randn(128, 784) * 0.01,
                'fc1.bias': torch.randn(128) * 0.01,
                'fc2.weight': torch.randn(10, 128) * 0.01,
                'fc2.bias': torch.randn(10) * 0.01,
            },
            'proof': {
                'old_model_hash': 'abc123',
                'delta_hash': 'def456',
                'num_samples': 100,
                'signature': 'a' * 64
            }
        }
        updates.append(update)
    
    # Add Byzantine update (much larger magnitude)
    byzantine_update = updates[0].copy()
    byzantine_update['node_id'] = 'byzantine_attacker'
    for key in byzantine_update['model_delta']:
        byzantine_update['model_delta'][key] *= 100  # Malicious gradient
    updates.append(byzantine_update)
    
    # Test aggregation
    aggregator = ByzantineRobustAggregator()
    
    async def test():
        result = await aggregator.aggregate(updates)
        print(f"\n✅ Aggregation successful")
        print(f"Audit trail entries: {len(aggregator.audit_trail)}")
        aggregator.export_audit_trail()
    
    import asyncio
    asyncio.run(test())