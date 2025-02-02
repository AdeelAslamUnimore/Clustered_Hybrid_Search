# Versatile Sketch-Based Attribute Filtering for Hybrid Vector Search

## Overview

This project introduces a **versatile approach** for hybrid vector search, where the **popularity of attributes** is considered to improve search efficiency. The proposed method utilizes a **two-hop search strategy** based on attribute popularity, allowing for an adaptable and scalable search process. By dynamically adjusting the search behavior based on the attributes and clusters, this method performs more efficiently on large-scale datasets. 

### Key Process Flow

1. **Graph Construction**: The approach begins with building an **HNSW graph** using specific parameters.
2. **Clustering**: A **two-hop search strategy** is applied to cluster the dataset.
3. **Sketch Construction**: In-memory data structures are built to store attribute predictors for each cluster.
4. **Querying**: The query is executed against the **HNSW graph** to find nearest neighbors.
5. **Filtering**: The **top-K nearest neighbors** are selected based on the search results.
6. **Cluster Analysis**: Each cluster’s popularity is evaluated, and based on the popularity, either a **two-hop search** or **one-hop search** is performed.

### Advantages of the Approach

- **Dynamic Search Adaptation**: The search method adjusts dynamically based on attribute popularity and the characteristics of the clusters.
- **Efficient Search**: By incorporating attribute popularity, the approach boosts the quality and speed of the search.
- **Scalability**: The approach is well-suited for large-scale datasets and works effectively with diverse metadata.

---

## Approach

The key steps of the proposed approach are:

1. **Graph Construction**: An HNSW graph is built using specific parameters such as `M` and `efConstruction`.
2. **Clustering**: The dataset is clustered using a **two-hop search strategy** applied to the HNSW graph at level 0.
3. **Sketch Construction**: In-memory predictor data structures are created to store attributes within each cluster, enhancing attribute filtering.
4. **Querying**: The query is executed by searching the **HNSW graph** to find the nearest neighbors.
5. **Filtering**: The **top-K nearest neighbors** are filtered, based on their relevance and search results.
6. **Cluster Analysis**: The popularity of each cluster is evaluated; if the popularity exceeds a threshold (`Δ`), a **two-hop search** is performed. Otherwise, a **one-hop search** is applied based on selectivity.

The approach adapts dynamically to attribute popularity and efficiently filters out irrelevant results, ensuring the search quality is maintained across different scenarios.

![ClusteredHNSW](https://raw.githubusercontent.com/AdeelAslamUnimore/Clustered_Hybrid_Search/main/ClusteringHNSW.png)

---

### Benefits of the Approach

By incorporating **attribute popularity** into the search process, this approach improves both **search quality** and **speed**. Additionally, it is **agnostic** to the specific attributes or metadata involved, making it versatile and applicable to a wide range of datasets and domains.
