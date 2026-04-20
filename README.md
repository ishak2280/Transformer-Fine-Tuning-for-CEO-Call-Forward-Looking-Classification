📊 Transformer Fine-Tuning for CEO Call Semantic Drift Analysis








🚀 Overview

This project builds an end-to-end NLP pipeline to analyze CEO earnings call transcripts by:

Extracting forward-looking statements
Applying LLM-based weak labeling
Fine-tuning a transformer sentence encoder
Tracking semantic drift over time (quarter-level trends)

Unlike traditional NLP tasks, this project focuses on improving embedding quality (representation learning) rather than prediction.

🎯 Problem Statement

Generic sentence embeddings fail to capture domain-specific meaning in financial text.

👉 This project answers:

Can we fine-tune embeddings so they reflect structured business signals like strategy, certainty, and market stance?

🧠 Approach
🔹 Phase 1: Weak Labeling (LLM-powered)
Model: Meta Llama 3 (via Ollama)
Extract only forward-looking sentences
Assign structured labels:
strategic_focus
temporal_framing
certainty_level
market_position
🔹 Phase 2: Transformer Fine-Tuning
Base Model: all-MiniLM-L6-v2
Objective: Label-aware embedding alignment
Method:
Contrastive learning
In-batch negatives
Label signature grouping

📈 Evaluation:

Loss convergence
Label alignment improvement
Clustering quality metrics
UMAP / t-SNE visualization
🔹 Phase 3: Semantic Drift Tracking
Clustering: KMeans (K=30)
Time-based aggregation:
Cluster volume (counts)
Cluster composition (shares)

📊 Output:

Topic evolution over quarters
Label-specific trend analysis (e.g., AI focus)
📂 Repository Structure
📦 ceo-semantic-drift-analysis
 ┣ 📁 data
 ┃ ┣ raw_transcripts/
 ┃ ┣ processed_sentences/
 ┃ ┗ labeled_data/
 ┣ 📁 notebooks
 ┃ ┣ 01_extraction_labeling.ipynb
 ┃ ┣ 02_finetuning.ipynb
 ┃ ┗ 03_drift_analysis.ipynb
 ┣ 📁 src
 ┃ ┣ data_processing.py
 ┃ ┣ labeling.py
 ┃ ┣ training.py
 ┃ ┣ clustering.py
 ┃ ┗ evaluation.py
 ┣ 📁 outputs
 ┃ ┣ embeddings/
 ┃ ┣ clusters/
 ┃ ┗ visualizations/
 ┣ requirements.txt
 ┗ README.md
📊 Dataset
Source: Apple earnings call transcripts
Timeframe: 2014–2025 (~40+ quarters)
Size: ~50,000 sentence-level records

Each record contains:

Sentence text
Quarter metadata
Weak labels (4 dimensions)
Embedding vectors (384-dim)
📈 Key Results

✅ Improved label-alignment after fine-tuning
✅ More coherent embedding space vs baseline
✅ Stable and interpretable clusters
✅ Clear semantic drift patterns over time

💡 Example Insight:

AI-related discussions become more concentrated into fewer semantic clusters in later years, indicating strategic consolidation.

🛠️ Tech Stack
Languages: Python
Deep Learning: PyTorch, Sentence Transformers
ML Tools: Scikit-learn
Visualization: UMAP, t-SNE
LLM Integration: Ollama (Llama 3)
Data: Pandas, NumPy
⚙️ Installation
git clone https://github.com/your-username/ceo-semantic-drift-analysis.git
cd ceo-semantic-drift-analysis

pip install -r requirements.txt
▶️ Usage
1. Run Weak Labeling
python src/labeling.py
2. Fine-Tune Model
python src/training.py
3. Run Drift Analysis
python src/clustering.py
🧪 Evaluation Metrics
Silhouette Score
Calinski-Harabasz Index
Davies-Bouldin Score
Label Alignment Metrics
⚠️ Limitations
Weak labels introduce noise
Single-company dataset limits generalization
Clustering is not ground truth
Sensitive to label imbalance
🔮 Future Work
Multi-company, multi-sector expansion
Domain-adaptive pretraining
Better label calibration
Dynamic clustering methods
📜 Disclaimer

This project is for educational purposes only and does not constitute financial or investment advice.

👩‍💻 Contributors
Isha Prakash Kadam
Dujun Zhai
Yifeng Qiu

Instructor: Christopher Dunham

⭐ If you found this useful

Give it a star ⭐ and feel free to connect!
