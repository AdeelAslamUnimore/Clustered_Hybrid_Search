# Versatile Sketch-Based Attribute Filtering for Hybrid Vector Search

## Overview

This project introduces a versatile approach for hybrid vector search that considers the popularity of attributes to improve search efficiency. By employing a two-hop search strategy based on attribute popularity, the method adapts the search behavior to better handle large-scale datasets. The process includes constructing an HNSW graph, performing clustering using a two-hop search, and building in-memory data structures for attributes in each cluster.

## Approach

The key steps of the proposed approach are:

1. **Graph Construction**: An HNSW graph is built using specific parameters.
2. **Clustering**: The dataset is clustered using a two-hop search strategy.
3. **Sketch Construction**: In-memory predictor data structures are created for attributes within each cluster.
4. **Querying**: The query is run against the HNSW graph.
5. **Filtering**: The top-K nearest neighbors are filtered based on the search results.
6. **Cluster Analysis**: The popularity of each cluster is evaluated, and if it exceeds a threshold, a two-hop search is performed. Otherwise, a one-hop search based on selectivity is conducted.

The approach ensures efficient and accurate results by dynamically adjusting the search method according to attribute popularity and cluster characteristics.

## Benefits

By incorporating attribute popularity into the search process, this approach delivers enhanced search quality and speed, moreover, it is agnostic for attributes and metadata. 
![Figure Description](ClusteringHNSW (3).eps)
