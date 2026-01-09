# 🛡️ Quantum Fortress

A **privacy-preserving federated learning prototype** with **integrity monitoring and attack detection**.

Built to explore how multiple parties can collaboratively train models **without sharing raw data**, while detecting anomalous or malicious behaviour during training.

---

## 🎯 What This Project Does

Quantum Fortress demonstrates a **realistic federated learning system** with:

- Local training at each participant
- Secure aggregation of model updates
- Basic cryptographic protections
- Anomaly and integrity checks during training rounds
- Clear system boundaries and observable behaviour

The focus is **deployability and system design**, not research novelty or marketing claims.

---

## 🧠 Why This Exists

Federated learning is often presented as “secure by default”.  
In practice, it is vulnerable to:

- Model poisoning
- Malicious or faulty participants
- Silent integrity failures
- Poor observability in production

This project explores **how those risks can be detected and surfaced**, rather than ignored.

---

## 🔍 Key Features

- **Federated Training Engine**
  - Multiple nodes train locally
  - Central aggregator combines updates

- **Attack / Anomaly Detection**
  - Flags suspicious model updates
  - Detects statistical deviations between rounds

- **Integrity & Auditability**
  - Training rounds are observable
  - Behaviour is logged and inspectable

- **Containerised Deployment**
  - Docker & docker-compose support
  - Designed to run locally or in cloud environments

---

## 🚀 Quick Start

### Docker (recommended)

```bash
git clone https://github.com/lesyed/QF-Protocol.git
cd QF-Protocol
docker-compose up
