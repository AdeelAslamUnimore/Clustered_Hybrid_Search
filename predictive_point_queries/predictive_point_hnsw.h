#pragma once
#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"
#include "../predictive_range_queries/ranges.h"
#include "count_min_sketch_min_hash.h"
#include "memory_access.h"
#include <vector>
#include <cstring>
#include <cstdint>
namespace clustered_hybrid_search
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    template <typename dist_t>
    class PredictivePointHNSW : public hnswlib::HierarchicalNSW<dist_t>
    {
        using Candidate = std::pair<dist_t, tableint>;
        // Comparator
        struct CompareByFirstElement
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };

    public:
        std::unordered_map<tableint, std::vector<std::string>> meta_data_predicates;
        std::unordered_map<unsigned int, CountMinSketchMinHash> mapForCMS;
        std::unordered_map<tableint, std::unordered_set<unsigned int>> cluster_Mem_chk;
        char *mem_for_ids_for_clusters{nullptr};
        char *bit_array_for_disk_access;
        char *filter_id_map;
        size_t max_elements_;

    public:
        PredictivePointHNSW(hnswlib::SpaceInterface<dist_t> *space, size_t max_elements, const std::string &location_of_index, std::unordered_map<tableint, std::vector<std::string>> &meta_data_predicates_)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements)
        {
            this->loadIndex(location_of_index, space, max_elements);
            meta_data_predicates = meta_data_predicates_;
            mem_for_ids_for_clusters = (char *)malloc(max_elements * (3 * sizeof(char)));
            memset(mem_for_ids_for_clusters, 0, max_elements * (3 * sizeof(char)));
            // Open the file in binary mode
            if (mem_for_ids_for_clusters == nullptr)
                throw std::runtime_error("Not enough memory");
            max_elements_ = max_elements;
        }
        // Method to perform clustering and maintain sketches for predictive point queries. It is actually conisder the two-hop neighbors for clustering
        // Here we employ Count Min Sketch with Min Hash for each cluster to maintain the sketches
        // We also can also consider the full keys which later we conider for entry points selection during search
        void clustering_and_maintaining_sketches(tableint size_of_cluster)
        {
            unsigned int clusterNumber = 0;
            int counterForFilter = 0;
            // Use single CMS instance instead of vector
            CountMinSketchMinHash current_cms;
            // For keeping track of visited nodes GLOBALLY
            std::unordered_set<tableint> visitedIds;
            std::unordered_set<tableint> visitedIdsLocally; // Keeping track of local visited Ids for a cluster

            for (tableint id = 0; id < meta_data_predicates.size(); id++)
            {
                if (visitedIds.find(id) != visitedIds.end())
                    continue;
                visitedIds.insert(id);

                // Update CMS for initial node // Second hop neighbors will be updated later this node 
                // for (const auto &predicate : meta_data_predicates[id])
                // {
                //     current_cms.update(predicate, id, 1);
                // }

                bit_manipulation_short(id, clusterNumber);
                counterForFilter++;

                int *data = (int *)this->get_linklist0(id);
                if (!data) // ✅ Add null check
                    continue;

                size_t size = this->getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop neighbors
                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);
                    if (visitedIdsLocally.find(candidateId) != visitedIdsLocally.end())
                        continue;
                    visitedIdsLocally.insert(candidateId);

                    // Update CMS
                    for (const auto &predicate : meta_data_predicates[candidateId])
                    {
                        current_cms.update(predicate, candidateId, 1);
                    }

                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                }

                // Two hop neighbors (separate loop like in your other code)
                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);

                    int *twoHopData = (int *)this->get_linklist0(candidateId);
                    if (!twoHopData) // ✅ Add null check
                        continue;

                    size_t twoHopSize = this->getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);

                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);

                        visitedIds.insert(candidateIdTwoHop);
                        if (visitedIdsLocally.find(candidateIdTwoHop) != visitedIdsLocally.end())
                            continue;
                        visitedIdsLocally.insert(candidateIdTwoHop);

                        // Update CMS
                        for (const auto &predicate : meta_data_predicates[candidateIdTwoHop])
                        {
                            current_cms.update(predicate, candidateIdTwoHop, 1);
                        }

                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);
                        counterForFilter++;
                    }
                }

                // Update data structure when cluster is full
                if (counterForFilter >= size_of_cluster)
                {
                    // ✅ Use move semantics
                    current_cms.total = counterForFilter;
                    mapForCMS[clusterNumber] = std::move(current_cms);

                    // Reset for next cluster
                    clusterNumber++;
                    counterForFilter = 0;
                    current_cms = CountMinSketchMinHash(); // Create new CMS
                    visitedIdsLocally.clear();             // Clear local visited IDs
                }
            }

            // Handle remaining nodes
            if (counterForFilter > 0)
            {
                current_cms.total = counterForFilter;
                mapForCMS[clusterNumber] = std::move(current_cms);
            }
        }

        // It also maintains pointers for disk optimizations
        void clustering_maintaining_sketches_with_disk_optimization(tableint size_of_cluster)
        {
            // Implement clustering and sketch maintenance logic with disk optimization here

            // Define the memory block for Disk optimization access
            unsigned int file_count = 3000000; // Block size for disk access
            unsigned int figureprint_size = 8;
            bit_array_for_disk_access = (char *)malloc(file_count * figureprint_size);
            if (!bit_array_for_disk_access)
            {
                std::cerr << "Memory allocation failed for bit array" << std::endl;
                return;
            }
            memset(bit_array_for_disk_access, 0, file_count * figureprint_size);

            std::vector<CountMinSketchMinHash> cms(1); // Initialize with one sketch
            unsigned int cms_counter = 0;
            unsigned int clusterNumber = 1;
            int counterForFilter = 0; // Used for cluster updates

            // To keep track of visited nodes
            std::unordered_set<tableint> visitedIds;

            for (tableint id = 0; id < meta_data_predicates.size(); id++)
            {
                if (visitedIds.find(id) != visitedIds.end())
                    continue; // Skip if already visited
                visitedIds.insert(id);
                // Inserting key and into disk optimized data structure
                compute_location(id, meta_data_predicates[id], file_count, figureprint_size);
                // Insert data into the sketch for the current node
                for (const auto &predicate : meta_data_predicates[id])
                {
                    cms[cms_counter].update(predicate, id, 1);
                }

                // Perform bit manipulations and update the cluster
                bit_manipulation_short(id, clusterNumber);
                counterForFilter++;

                // One-hop insertion (linked nodes of current id)
                int *data = (int *)this->get_linklist0(id);
                size_t size = this->getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
                std::unordered_set<tableint> localVisitedIds; // Keeping track of local visited Ids for a cluster

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    if (localVisitedIds.find(candidateId) != localVisitedIds.end())
                        continue;
                    localVisitedIds.insert(candidateId);
                    visitedIds.insert(candidateId);
                    compute_location(candidateId, meta_data_predicates[candidateId], file_count, figureprint_size);

                    for (const auto &predicate : meta_data_predicates[candidateId])
                    {
                        cms[cms_counter].update(predicate, candidateId, 1);
                    }

                    // Update clusters and bitmaps

                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;

                    // Two-hop insertion (linked nodes of the candidate node)
                    int *twoHopData = (int *)this->get_linklist0(candidateId);
                    size_t twoHopSize = this->getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);

                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        if (localVisitedIds.find(candidateIdTwoHop) != localVisitedIds.end())
                            continue;
                        localVisitedIds.insert(candidateIdTwoHop);
                        visitedIds.insert(candidateIdTwoHop);
                        compute_location(candidateIdTwoHop, meta_data_predicates[candidateIdTwoHop], file_count, figureprint_size);
                        for (const auto &predicate : meta_data_predicates[candidateIdTwoHop])
                        {
                            cms[cms_counter].update(predicate, candidateIdTwoHop, 1);
                        }

                        // Update clusters and bitmaps

                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);
                        counterForFilter++;
                    }
                }

                // After processing current cluster, check if we need to save the cluster
                if (counterForFilter >= size_of_cluster)
                {
                    mapForCMS[clusterNumber] = std::move(cms[cms_counter]); // Store the CMS
                    clusterNumber++;
                    cms_counter++;
                    cms.push_back(CountMinSketchMinHash()); // Push a new sketch
                    counterForFilter = 0;
                }
            }

            // Handle any remaining data
            if (counterForFilter > 0)
            {
                mapForCMS[clusterNumber] = std::move(cms[cms_counter]);
            }
        }

        /**
         * @brief Updates a node's cluster information using bit manipulation.
         * Reads 3 bytes of data, checks if the least significant bit is set, and either updates the value or inserts a new cluster.
         * The function modifies the `mem_for_ids_clusters` and updates the `cluster_Mem_chk` map accordingly.
         */

        void bit_manipulation_short(const int &node_identifier_, const unsigned int &cluster_identifier_)
        {
            // size_t byte_offset = 3;
            uint32_t value_at_index = 0;
            memcpy(&value_at_index, mem_for_ids_for_clusters + (node_identifier_ * 3), 3);

            // Check the first bit (least significant bit)
            bool is_first_bit_set = (value_at_index & 1) != 0;

            if (is_first_bit_set)
            {
                //  cout<<"is irst bit iniside"<<value_at_index<<endl;
                unsigned int clusterID_ = value_at_index >> 8; // for Map Insertion when it contain more values

                // Update the hashvalue or maintain the hashvalue for clusters.
                value_at_index = value_at_index | (1 << 1);

                // value_at_index=value_at_index|(1 << clusterID);
                memcpy(mem_for_ids_for_clusters + (node_identifier_ * 3), &value_at_index, 3);

                // Update map.

                if (cluster_Mem_chk.find(node_identifier_) != cluster_Mem_chk.end())
                {

                    // If present, add the clusterNumber to the set
                    cluster_Mem_chk[node_identifier_].insert(cluster_identifier_);
                }
                else
                {
                    std::unordered_set<unsigned int> newSet;
                    newSet.insert(clusterID_);          // First id
                    newSet.insert(cluster_identifier_); // Second ID
                    cluster_Mem_chk[node_identifier_] = newSet;
                }
            }
            else
            {
                value_at_index |= (cluster_identifier_ << 8);
                value_at_index |= (1 << 0); // set on position 0 if the array position is 0
                memcpy(mem_for_ids_for_clusters + (node_identifier_ * 3), &value_at_index, 3);
            }
        }

        // computing the location and setting the respective on
        // Here figureprint should be 8, 16, 32, 64 bit
        void compute_location(
            unsigned int &id,
            const std::vector<std::string> &predicates,
            unsigned int file_count,
            int size_of_figureprint)
        {
            constexpr int NUM_HASH = 3;
            constexpr int BASE_SEED = 45;

            for (int i = 0; i < NUM_HASH; ++i)
            {

                uint32_t hash_value =
                    MurmurHash64B(&id, sizeof(id), BASE_SEED + i) % file_count;

                size_t index = hash_value / size_of_figureprint;

                for (const std::string &predicate : predicates)
                {

                    uint32_t bit_pos =
                        MurmurHash64B(
                            predicate.data(),
                            predicate.size(),
                            BASE_SEED + i) %
                        size_of_figureprint;

                    bit_array_for_disk_access[index] |= (1ULL << bit_pos);
                }
            }
        }

        void testing_function()
        {
            std::cout << "Testing function called." << mapForCMS.size() << std::endl;

            // Check if map has data
            if (mapForCMS.empty())
            {
                std::cout << "ERROR: mapForCMS is still empty after clustering!" << std::endl;
                return;
            }

            std::cout << "Total clusters in map: " << mapForCMS.size() << std::endl;

            // Retrieve the FIRST CMS from the map
            // Method 1: Using begin() iterator
            auto first_cluster_it = mapForCMS.begin();
            unsigned int first_cluster_id = first_cluster_it->first;
            CountMinSketchMinHash &first_cms = first_cluster_it->second;

            std::cout << "\n--- First Cluster Info ---" << std::endl;
            std::cout << "Cluster ID: " << first_cluster_id << std::endl;
            std::cout << "Total count: " << first_cms.totalcount() << std::endl;

            // Predict/Estimate for string "4"
            std::string query = "2";

            std::cout << "\n--- Predicting for string: \"" << query << "\" ---" << std::endl;

            // Get the estimate (returns pair of indices)
            auto result = first_cms.estimate(query);
            unsigned int min_j = result.first;        // Row index (which hash function)
            unsigned int min_hashval = result.second; // Column index (which bucket)

            std::cout << "Hash function index (j): " << min_j << std::endl;
            std::cout << "Bucket index (hashval): " << min_hashval << std::endl;

            // Get the actual count from the CMS
            unsigned int estimated_count = first_cms.C[min_j][min_hashval];
            std::cout << "Estimated count for \"" << query << "\": " << estimated_count << std::endl;

            // Get the MinHash set for this bucket
            const auto &minhash_set = first_cms.full_keys[min_j][min_hashval];
            std::cout << "MinHash set size: " << minhash_set.size() << std::endl;

            // Implement testing logic here
        }
        // Search function for predictive point queries

        void predicateCondition(char *filters_array)
        {
            filter_id_map = filters_array;
        }

        void search(const void *query_data, size_t query_number, const std::vector<std::string> &query_predicates, size_t top_k)
        {
            std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement> result_queue = predictive_point_query(query_data, query_number, query_predicates, top_k);
           // std::cout << "Results for query number " << query_number << " :" << result_queue.size() << std::endl;
        }

        std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement> predictive_point_query(const void *query_data, size_t query_number, const std::vector<std::string> &query_predicates, size_t top_k)
        {
          
                // Map for removing overhead of duplicate distance computation
                std::unordered_map<tableint, dist_t> distance_map;

                // Get search results using post filtering
                auto search_results = this->searchKnnForPredictiveStructures(query_data, top_k, &distance_map);

                // Priority queue for the results
                std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement> result_queue;

                // Pre-allocate with estimated size
                const size_t estimated_size = search_results.size();
                std::unordered_set<tableint> visitedIds;
                visitedIds.reserve(estimated_size);

                std::unordered_set<tableint> visitedClusters;
                visitedClusters.reserve(estimated_size / 10); // Estimate fewer clusters than nodes

                std::vector<tableint> indices;
                indices.reserve(estimated_size);

                const size_t filter_offset = query_number * max_elements_;
                const size_t result_limit = top_k * 2;
                const double popularity_threshold = 0.50;

                // Phase 1: Process initial search results
                while (!search_results.empty())
                {
                    auto [dist, id] = search_results.top();
                    search_results.pop();

                    visitedIds.insert(id);

                    if (filter_id_map[filter_offset + id])
                        result_queue.push({dist, id});

                    indices.push_back(id);
                }

                // Early exit if we have enough results
                if (result_queue.size() > top_k)
                {
                    if (result_queue.size() > result_limit)
                        return result_queue;

                    // One-hop expansion for all indices
                    for (tableint id : indices)
                    {
                        one_hop_search_neighbors(query_data, id, query_number,
                                                 visitedIds, distance_map, result_queue);

                        if (result_queue.size() > result_limit)
                            return result_queue;
                    }
                    return result_queue;
                }

                // Phase 2: Cluster-based expansion
                for (tableint id : indices)
                {
                    auto clusters = cluster_contains_attribute(id);
                    std::cout<<"Clusters size: "<<clusters.size()<<std::endl;
                    auto [cluster_id, max_pop] = popularity_computation(clusters, query_predicates);

                    // Check if cluster already visited
                    if (!visitedClusters.insert(cluster_id).second)
                    {
                        // Cluster was already visited, do one-hop
                        one_hop_search_neighbors(query_data, id, query_number,
                                                 visitedIds, distance_map, result_queue);
                    }
                    else
                    {
                        // New cluster - choose expansion strategy based on popularity
                        if (max_pop > popularity_threshold)
                        {
                            two_hop_search_neighbors(query_data, id, query_number,
                                                     visitedIds, distance_map, result_queue);
                        }
                        else
                        {
                            one_hop_search_neighbors(query_data, id, query_number,
                                                     visitedIds, distance_map, result_queue);
                        }
                    }

                    // Early exit on size limit
                    if (result_queue.size() > result_limit)
                        return result_queue;
                }

                return result_queue;
            }
            // Popularity prediction function for predictive point queries
            inline std::vector<unsigned int> cluster_contains_attribute(hnswlib::labeltype id) const
            {
                // Decode cluster info
                union Data data;
                const size_t offset = static_cast<size_t>(id) * 3;

                data.raw.bytes[0] = mem_for_ids_for_clusters[offset];
                data.raw.bytes[1] = mem_for_ids_for_clusters[offset + 1];
                data.raw.bytes[2] = mem_for_ids_for_clusters[offset + 2];
                data.raw.padding = 0;

                const bool is_multi_cluster = (data.value & (1 << 1)) != 0;
                const uint32_t primary_cluster = data.value >> 8;

                // Single-cluster case (fast path)
                if (!is_multi_cluster)
                {
                    return {primary_cluster};
                }

                // Multi-cluster case
                const auto &clusters = cluster_Mem_chk.at(id);

                std::vector<unsigned int> result;
                result.reserve(clusters.size()); //

                for (unsigned int cid : clusters)
                {
                    result.push_back(cid);
                }

                return result;
            }
            // Popularity computation function for predictive point queries
            std::pair<size_t, double> popularity_computation(std::vector<unsigned int> & clusters, const std::vector<std::string> &query_predicates)
            {
                double max_popularity = 0.0; // this will store the maximum fraction within any cluster
                size_t cluster_id = 0;
                for (const auto &cid : clusters)
                {
                    auto cms_it = mapForCMS.find(cid);
                    if (cms_it != mapForCMS.end())
                    {
                        CountMinSketchMinHash &cms = cms_it->second;

                        // Estimate count for the predicate
                        auto estimate_result = cms.estimate(query_predicates[0]);
                        unsigned int estimated_count = cms.C[estimate_result.first][estimate_result.second];

                        // Compute popularity inside this cluster
                        unsigned int cluster_total = cms.totalcount();
                        double popularity = cluster_total > 0 ? static_cast<double>(estimated_count) / cluster_total : 0.0;
                        // Keep the cluster with the max popularity
                        if (popularity > max_popularity)
                        {
                            max_popularity = popularity;
                            cluster_id = cid;
                        }
                    }
                }
                return {cluster_id, max_popularity};
            }
            // Use it when we use for disk access optimization or use the allocated memory deletion in the destructor
            // One hop Search neighbors
            void one_hop_search_neighbors(const void *data_point, tableint id, size_t query_number, std::unordered_set<tableint> &visitedIds, std::unordered_map<tableint, dist_t> &distance_map, std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement> &result_queue)
            {
                int *data = (int *)this->get_linklist0(id);
                size_t size = this->getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop neighbors
                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    if (visitedIds.find(candidateId) != visitedIds.end())
                        continue;
                    visitedIds.insert(candidateId);
                    if (!filter_id_map[query_number * max_elements_ + candidateId])
                        continue;

                    dist_t distance;
                    // Check if distance has already been computed during initial search
                    auto dist_it = distance_map.find(candidateId);
                    if (dist_it != distance_map.end())
                    {
                        distance = dist_it->second;
                        result_queue.push({distance, candidateId});
                    }
                    else
                    {
                        distance = this->fstdistfunc_(data_point, this->getDataByInternalId(candidateId), this->dist_func_param_);
                        result_queue.push({distance, candidateId});
                    }
                }
            }
            // Two hop Search neighbors
            void two_hop_search_neighbors(
                const void *data_point,
                tableint start_id,
                size_t query_number,
                std::unordered_set<tableint> &visitedIds,
                std::unordered_map<tableint, dist_t> &distance_map,
                std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement> &result_queue)
            {
                // ---------------- First hop ----------------
                int *data1 = (int *)this->get_linklist0(start_id);
                if (!data1)
                    return;

                size_t size1 = this->getListCount((linklistsizeint *)data1);
                tableint *datal1 = (tableint *)(data1 + 1);

                for (size_t i = 0; i < size1; i++)
                {
                    tableint firstHopId = datal1[i];

                    // ---------- Distance handling for first hop ----------
                    if (visitedIds.find(firstHopId) == visitedIds.end() &&
                        filter_id_map[query_number * max_elements_ + firstHopId])
                    {
                        visitedIds.insert(firstHopId);
                        auto it1 = distance_map.find(firstHopId);
                        dist_t dist1 = (it1 != distance_map.end())
                                           ? it1->second
                                           : this->fstdistfunc_(
                                                 data_point,
                                                 this->getDataByInternalId(firstHopId),
                                                 this->dist_func_param_);

                        result_queue.push({dist1, firstHopId});
                    }
                    // ---------------- Second hop ----------------
                    int *data2 = (int *)this->get_linklist0(firstHopId);
                    if (!data2)
                        continue;

                    size_t size2 = this->getListCount((linklistsizeint *)data2);
                    tableint *datal2 = (tableint *)(data2 + 1);

                    for (size_t j = 0; j < size2; j++)
                    {
                        tableint secondHopId = datal2[j];

                        if (visitedIds.find(secondHopId) != visitedIds.end())
                            continue;
                        visitedIds.insert(secondHopId);

                        if (!filter_id_map[query_number * max_elements_ + secondHopId])
                            continue;

                        auto it2 = distance_map.find(secondHopId);
                        dist_t dist2 = (it2 != distance_map.end())
                                           ? it2->second
                                           : this->fstdistfunc_(
                                                 data_point,
                                                 this->getDataByInternalId(secondHopId),
                                                 this->dist_func_param_);

                        result_queue.push({dist2, secondHopId});
                    }
                }
            }

            void freeMemory()
            {
                if (mem_for_ids_for_clusters)
                {
                    free(mem_for_ids_for_clusters);
                    mem_for_ids_for_clusters = nullptr;
                }
                if (bit_array_for_disk_access)
                {
                    free(bit_array_for_disk_access);
                    bit_array_for_disk_access = nullptr;
                }
            }

            ~PredictivePointHNSW()
            {
                if (mem_for_ids_for_clusters)
                {
                    free(mem_for_ids_for_clusters);
                    mem_for_ids_for_clusters = nullptr;
                }
            };
        };
    }