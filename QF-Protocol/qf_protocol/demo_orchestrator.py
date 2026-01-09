# demo_orchestrator.py
import asyncio
import torch
from core.federated_engine.node import SecureFederatedNode, TrainingConfig
from core.federated_engine.aggregator import ByzantineRobustAggregator

async def main():
    print("🚀 Launching QF-Protocol Demo...")
    
    # 1. Setup Aggregator
    aggregator = ByzantineRobustAggregator()
    
    # 2. Setup Nodes (Simulating 3 Hospitals)
    nodes = [
        SecureFederatedNode(TrainingConfig(node_id="St_Jude_Hospital")),
        SecureFederatedNode(TrainingConfig(node_id="Mayo_Clinic")),
        SecureFederatedNode(TrainingConfig(node_id="Malicious_Actor_Node")) # We'll simulate an attack
    ]
    
    # 3. Generate Synthetic Data
    X = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))

    # 4. Run a simulated round
    print("🛰️  Nodes starting local training...")
    updates = []
    for node in nodes:
        update = await node.train_round(X, y, None)
        
        # Simulate a Byzantine attack from the 3rd node
        if node.config.node_id == "Malicious_Actor_Node":
            for key in update['model_delta']:
                update['model_delta'][key] *= 100 # Sabotage!
        
        updates.append(update)

    # 5. Aggregate & Defend
    print("🛡️  Aggregator analyzing updates...")
    global_model = await aggregator.aggregate(updates)
    
    # 6. Check Audit Trail
    aggregator.export_audit_trail("logs/hospital_demo_audit.json")

if __name__ == "__main__":
    asyncio.run(main())