# Week 4 — Unsupervised Learning & Customer Segmentation

Internship task (ITSimplera) applying **K-Means** and **Hierarchical (Agglomerative)
Clustering** to segment ~9,000 credit card holders into behavioural groups, using the
[Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

## 🎯 Objective

Unlike prior supervised-learning weeks, this task has no target column. The goal is to
discover natural customer groupings from 17 behavioural features (balance, purchases,
cash advances, credit limit, payment history) so a bank could use the segments for
targeted marketing, credit risk management, and personalized services.

## 📁 Repository Structure

```
.
├── data/
│   └── CC_GENERAL.csv              # raw dataset
├── notebooks/
│   └── week4_clustering.ipynb      # full analysis, all cells run with outputs
├── outputs_elbow_curve.png         # k vs inertia
├── outputs_silhouette_scores.png   # k vs silhouette score
├── outputs_cluster_heatmap.png     # cluster feature-mean heatmap
├── outputs_dendrogram.png          # hierarchical clustering dendrogram
├── outputs_crosstab_heatmap.png    # K-Means vs Hierarchical agreement
├── requirements.txt
└── README.md
```

## 🔍 What's inside the notebook

**Part 1 — K-Means Clustering**

1. Load data, drop `CUST_ID`, inspect shape/dtypes
2. Handle missing values (median imputation for `MINIMUM_PAYMENTS`, `CREDIT_LIMIT`)
3. Scale all features with `StandardScaler`
4. Run K-Means for k = 2–10, track inertia
5. Elbow curve to find the bend point
6. Silhouette score per k, plotted and compared against the elbow method
7. Final K-Means fit at the chosen optimal k
8. Cluster profiling (mean feature values) + heatmap
9. Business-language interpretation of each cluster

**Part 2 — Hierarchical Clustering**

1. Agglomerative clustering (`scipy`, Ward linkage) on a random 300-row sample
2. Dendrogram with a labeled cut-threshold line
3. `AgglomerativeClustering` (scikit-learn) with the same k as Part 1
4. Cross-tabulation comparing K-Means vs Hierarchical cluster assignments
5. Written comparison: interpretability, meaningfulness, and production recommendation

## 🧠 Key Findings

- Optimal number of clusters: **k = 4**, selected via the elbow method and cross-checked
  against silhouette scores.
- Four business-interpretable segments emerged: **cash-advance users** (higher risk),
  **high-value spenders** (most profitable), **low-engagement/dormant customers**
  (re-engagement targets), and **responsible installment shoppers** (low-risk, steady
  revenue).
- K-Means and Hierarchical Clustering largely agree on the sample cross-tabulation,
  validating the segmentation. **K-Means is recommended for production** due to its
  scalability to the full customer base; hierarchical clustering is best used as a
  periodic validation/exploration tool.

## ⚙️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/week4_clustering.ipynb
```

## 🛠 Tech Stack

Python · pandas · numpy · scikit-learn · scipy · matplotlib · seaborn

## 👤 Author

Anwar Ul Haq — AI/ML Intern, ITSimplera
BS Artificial Intelligence, University of Agriculture Peshawar
