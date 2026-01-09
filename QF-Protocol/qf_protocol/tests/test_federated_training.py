import asyncio
import torch
from core.federated_engine.node import SecureFederatedNode, TrainingConfig
from core.federated_engine.aggregator import ByzantineRobustAggregator

async def main():
    print("🚀 Starting Quantum-Fortress Simulation")
    
    # 1. Setup
    aggregator = ByzantineRobustAggregator()
    nodes = [SecureFederatedNode(TrainingConfig(node_id=f"hospital_{i}")) for i in range(3)]
    
    # 2. Training Loop
    for round_num in range(3):
        print(f"\n--- ROUND {round_num} ---")
        updates = []
        
        for node in nodes:
            # Synthetic Data
            X, y = torch.randn(100, 784), torch.randint(0, 10, (100,))
            
            # Use global model if it exists
            update = await node.train_round(X, y, aggregator.global_model)
            updates.append(update)
            
        # Aggregation with Byzantine checks
        await aggregator.aggregate(updates)
    
    # 3. Finalize
    aggregator.export_audit_trail("logs/audit_trail.json")
    print("\n🎉 Simulation Finished. Audit trail saved to logs/audit_trail.json")

if __name__ == "__main__":
    asyncio.run(main())