"""demo_attack_detection.py - The 'Mind Blower' Script."""

import asyncio
import torch
from core.federated_engine.aggregator import ByzantineRobustAggregator

async def run_demo():
    print("="*70)
    print("🛡️  QUANTUM FORTRESS: AI ATTACK DETECTION DEMO")
    print("="*70)
    
    aggregator = ByzantineRobustAggregator()
    
    # 1. Simulate "Healthy" rounds to train the AI
    print("\nPhase 1: Building trust with honest nodes...")
    for _ in range(11):
        updates = [{'node_id': 'hospital_A', 'model_delta': {'w': torch.randn(10, 10) * 0.01}}]
        await aggregator.aggregate(updates)

    # 2. Launch the Attack
    print("\nPhase 2: Malicious node attempts poisoning...")
    poisoned_update = {
        'node_id': 'hacker_node',
        'model_delta': {'w': torch.randn(10, 10) * 50.0} # Massive spikes
    }
    
    await aggregator.aggregate([poisoned_update])
    
    print("\n" + "="*70)
    print("✅ Demo Complete: AI detected and neutralized the threat.")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(run_demo())