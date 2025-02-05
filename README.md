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
The approach adapts dynamically to attribute popularity and efficiently filters out irrelevant results, ensuring the search quality is maintained across different scenarios.

<p align="center">
  <img src="https://raw.githubusercontent.com/AdeelAslamUnimore/Clustered_Hybrid_Search/main/ClusteringHNSW.png" alt="ClusteredHNSW">
</p>

---

### Benefits of the Approach

By incorporating **attribute popularity** into the search process, this approach improves both **search quality** and **speed**. Additionally, it is **agnostic** to the specific attributes or metadata involved, making it versatile and applicable to a wide range of datasets and domains.
## **How to Run the Code**

### **1. Initial Setup**
Ensure a clean build environment before running the code:

```sh
rm -rvf ./build  # Remove existing build folder if it exists
mkdir build      # Create a new build folder
cd build         # Move into the build directory
```
### **2. Build the Code**
Inside the `build` folder, use CMake to compile the proposed method:

```
cmake -DUSE_ARRAY=ON -DUSE_BLOOM_FILTER=OFF -DUSE_CQF=OFF ..
```
### 3. Configuration

Once the code is compiled, navigate to the `examples/cpp` folder and locate the `constants_and_filepaths.txt` file. This file contains all the parameters for HNSW and the proposed clustering approach. Adjust the parameters according to your requirements. Below are some key parameters you may need to configure:

| **Parameter**               | **Value**                |
|-----------------------------|--------------------------|
| DIM                         | 768                      |
| M                           | 256                      |
| EFC                         | 400                      |
| CLUSTERSIZE                 | 1,000,000                |
| POPULARITYTHRESHOLDPOINT    | 2000                     |
| INTERSECTIONSIZE            | 250                      |
| POPULARITYTHRESHOLDCDF      | 0.4                      |
| INDEXPATH                   | /home/data/indexpath.bin |
| METADATAPATH                | /home/data/metadata.csv  |
| QUERIESPATH                 | /home/data/queries.csv   |
| RESULTFOLDER                | /home/data/              |
| DATASETFILE                 | /home/data/data_set.csv  |

### 4. Run Index Construction
To construct the index structure, you can run the following example:
```
./index_creation.cpp
```
### 5. Run Point Queries
To run the code for point queries, use the following:
 ```
./point_query_example.cpp
 ```
### 6. Recall Computation

For recall computation, refer to the notebook `Result_note_book.ipynb`. Be sure to update the paths and the ground truth computed results, along with the folder path where the results will be saved.

## Need Help or Have Questions?

If you encounter any issues while running or have questions about the project, feel free to reach out for assistance:

**Contact:**
- Adeel Aslam
- Email: [adeel.aslam@unimore.it](mailto:adeel.aslam@unimore.it)
