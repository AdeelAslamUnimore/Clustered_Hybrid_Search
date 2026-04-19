# 📘 Project README

> A step-by-step guide for building, indexing, querying, and evaluating the system.

---

## 📋 Table of Contents

- [🔨 Build Instructions](#-build-instructions)
- [📊 Result Computer](#-result-computer)
- [🗂️ Index Construction](#️-index-construction)
- [✅ Ground Truth Computation](#-ground-truth-computation)
- [🔍 Point Query Execution](#-point-query-execution)
- [🔍 Range Query Execution](#-range-query-execution)
- [🔍 Conjunction_Query Execution](#-conjunction-query-execution)
- [⚡ High Selectivity Query](#-high-selectivity-query)
- [📬 Contact](#-contact)

---

## 🔨 Build Instructions

```bash
# Step 1: Remove existing build directory
rm -rvf Build

# Step 2: Create a fresh build folder
mkdir Build

# Step 3: Navigate into it
cd Build

# Step 4: Configure with CMake
cmake ..

# Step 5: Compile
make
```

---

## 📊 Result Computer

> **⚙️ Usage Notes:**
> - Use the folder /examples/python_notebook/Recall_compute.ipynb
> - Update the folder paths (`ground_truth_folder`, `algorithm_base_folder`, `output_base_folder`) to match your local dataset and result directories **before running**.
> - Ensure result folders are named according to EFS values (e.g., `20`, `40`, `100`, ...).
> - Run the script to compute recall and generate summary results.

---

## 🗂️ Index Construction

> **▶️ Execution Instructions:**
> 1. Update the paths in `/example/constants/index_construction.txt` (e.g., `DATASET_FILE`, `INDEX_PATH`) to match your system.
> 2. Compile the code using CMake or g++ to generate the executable `index_construction`.
> 3. Run the executable:
>    ```bash
>    ./index_construction
>    ```
> 4. The program will read the dataset, build the HNSW index, and save it to `INDEX_PATH`.
> 5. ⚠️ Make sure the dataset file exists and has the correct format before execution.

---

## ✅ Ground Truth Computation

> **▶️ Execution Instructions:**
> 1. Update all paths in `/example/constants/ground_truth_batched_filtered.txt`:
>    `INDEX_PATH`, `META_DATA_PATH`, `QUERIES_PATH`, `FILTER_PATH`, `GROUND_TRUTH`
> 2. Ensure the HNSW index is already built and available at `INDEX_PATH` before running.
> 3. Run the program:
>    ```bash
>    ./example_filter_ground_truth_computer
>    ```
> 4. 🗃️ Filter maps will be cached in `FILTER_PATH` to avoid recomputation and improve performance.
> 5. 🚨 **IMPORTANT:** Keep `BATCH_OF_QUERIES` and `FILTER_PATH` consistent across **all** experiments. The filter can be computed once and reused for all runs.
> 6. This consistency ensures reproducibility and alignment with **ACON** and **NaviX** experimental setups.

---

## 🔍 Point Query Execution

> **📝 Notes:**
> 1. All parameters are defined and explained in the configuration file `point_query.txt`.
> 2. Refer to the descriptions above each parameter for their meaning and usage.
> 3. Modify values to control clustering, filtering, `efSearch` (EFS), and execution behaviour.
> 4. ⚠️ Ensure paths and parameters — especially `cluster_size` and `selectivity` — are set correctly before running.
> 5. Run using:
>    ```bash
>    ./example_point_search
>    ```

---

## ⚡ High Selectivity Query

> **📝 Notes:**
> 1. EFS values are used to select the **top-k clusters** based on attribute frequency.
> 2. `CLUSTER_SIZE` plays a critical role — it determines how data is partitioned and directly affects cluster quality.
> 3. Run using:
>    ```bash
>    ./example_cluster_level_search
>    ```


---

## 🔍 Range Query Execution

> **📝 Notes:**

> 1. Run using:
>    ```bash
>    ./example_range_search.cpp
>    ```


---

## 🔍 Disjunction Query Execution

> **📝 Notes:**

>1. Run using:
>    ```bash
>    ./example_disjunction_search.cpp
>    ```


---

## 🔍 Conjunction Query Execution

> **📝 Notes:**

>1. Run using:
>    ```bash
>    ./example_conjunction_search.cpp
>    ```


---

## 📬 Contact

If you encounter any issues or need assistance while running this code, feel free to reach out or open an issue:

**A. Aslam**  
📧 [A.Aslam@soton.ac.uk](mailto:A.Aslam@soton.ac.uk)

> _Don't hesitate to get in touch — happy to help!_ 🙂
