Transformer Fine-Tuning for CEO Call Forward-Looking Classification and Drift Tracking
Overview

This project builds an end-to-end deep learning pipeline to analyze CEO earnings call transcripts by extracting forward-looking statements, fine-tuning transformer-based sentence embeddings, and tracking semantic drift over time.

The primary goal is representation learning, not prediction. We improve sentence embeddings so they better reflect domain-specific semantic structures derived from weak labels.

Key Features
Weak labeling of forward-looking sentences using LLMs
Transformer fine-tuning for label-aware embeddings
Clustering-based topic modeling
Quarter-level semantic drift tracking
Label-aware analysis (focus, timeframe, certainty, stance)
Pipeline Architecture
Phase 1: Forward-Looking Sentence Extraction & Weak Labeling
Extract forward-looking sentences using an LLM (Meta Llama 3 via Ollama)
Assign structured labels:
strategic_focus
temporal_framing
certainty_level
market_position
Output: Clean, labeled sentence-level dataset
Phase 2: Transformer Fine-Tuning
Base model: sentence-transformers/all-MiniLM-L6-v2
Objective: Improve embedding alignment with label structure
Method:
Contrastive learning using label signatures
In-batch negative sampling
Evaluation:
Loss convergence
Label-alignment improvement
Clustering quality (Silhouette, CH, DB)
UMAP / t-SNE visualization
Phase 3: Clustering & Semantic Drift Tracking
KMeans clustering (K=30) on embeddings
Quarter-wise aggregation:
Cluster counts (volume)
Cluster shares (composition)
Label-specific drilldowns (e.g., AI focus trends)
Dataset
Source: Apple earnings call transcripts (2014–2025)
Size:
~40+ quarters
~50,000 sentence-level records
Features:
Sentence text
Quarter metadata
Weak labels (4 dimensions)
Embeddings (384-dimensional)
Results
Improved label alignment after fine-tuning
More coherent embedding space vs baseline
Stable and interpretable clusters
Clear semantic drift patterns across quarters
Example insight:
AI-related discussions become more concentrated into fewer clusters over time
Tech Stack
Python
PyTorch / Sentence Transformers
Scikit-learn (KMeans, metrics)
UMAP & t-SNE for visualization
Ollama (LLM inference)
Pandas / NumPy
Key Learnings
Weak labels can effectively guide representation learning
Fine-tuning improves semantic structure even without explicit prediction tasks
Embeddings can be used to track temporal shifts in meaning, not just similarity
Label-aware evaluation is more useful than generic clustering metrics
Limitations
Weak labels introduce noise
Single-company dataset limits generalization
Clustering is not ground truth
Sensitive to label imbalance and prompt design
Future Work
Extend to multi-company, multi-sector datasets
Improve label calibration and validation
Explore domain-adaptive pretraining
Experiment with dynamic clustering over time
