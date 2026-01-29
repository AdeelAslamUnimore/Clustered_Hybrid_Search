#pragma once
#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"
#include "../predictive_range_queries/ranges.h"
#include "../predictive_point_queries/memory_access.h"
#include "../predictive_point_queries/count_min_sketch_min_hash.h"
#include "../predictive_range_queries/regression.h"
#include <vector>
#include <cstring>
#include <cstdint>
#include <map>
#include <omp.h>
namespace clustered_hybrid_search
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    template <typename dist_t>
    class PredictiveConjuctiveHNSW : public hnswlib::HierarchicalNSW<dist_t>
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
        std::unordered_map<tableint, std::pair<std::vector<std::string>, int>> meta_data_predicates;
        std::unordered_map<unsigned int, CountMinSketchMinHash> mapForCMS;
        std::unordered_map<unsigned int, RegressionModel> mapForRegressionModel;
        std::unordered_map<tableint, std::unordered_set<unsigned int>> cluster_Mem_chk;
        char *mem_for_ids_for_clusters{nullptr};
        char *filter_id_map;
        size_t max_elements_;

    public:
        PredictiveConjuctiveHNSW(hnswlib::SpaceInterface<dist_t> *space, size_t max_elements, const std::string &location_of_index, std::unordered_map<tableint, std::pair<std::vector<std::string>, int>> &meta_data_predicates_)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements)
        {

            this->loadIndex(location_of_index, space, max_elements);
            meta_data_predicates = meta_data_predicates_;
            mem_for_ids_for_clusters = (char *)malloc(max_elements * (3 * sizeof(char)));
            memset(mem_for_ids_for_clusters, 0, max_elements * (3 * sizeof(char)));
            max_elements_ = max_elements;
            // Open the file in binary mode
            if (mem_for_ids_for_clusters == nullptr)
                throw std::runtime_error("Not enough memory");
        }

        void clustering_for_cms_and_cdf_filtering(tableint size_of_cluster)
        {
            // Implement clustering and sketch maintenance logic here

            unsigned int clusterNumber = 0;
            std::vector<std::pair<tableint, int>> predicate_data_CDF; // This is used for computing the CDF for range filtering
            CountMinSketchMinHash current_cms;
            int counterForFilter = 0;
            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;
            std::unordered_set<tableint> localVisitedIds;

            for (tableint id = 0; id < max_elements_; id++)
            {
                if (visitedIds.find(id) != visitedIds.end())
                    continue;
                visitedIds.insert(id);
                bit_manipulation_short(id, clusterNumber);
                counterForFilter++;
                int *data = (int *)this->get_linklist0(id);
                if (!data)
                    continue; // Error handling
                size_t size = this->getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
                // Keeping track of local visited Ids for a cluster
                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    if (localVisitedIds.find(candidateId) != localVisitedIds.end())
                        continue;
                    localVisitedIds.insert(candidateId);
                    visitedIds.insert(candidateId);

                    // Inserting the CDF Value here
                    // Update CMS
                    for (const auto &predicate : meta_data_predicates[candidateId].first)
                    {
                        current_cms.update(predicate, candidateId, 1);
                    }

                    predicate_data_CDF.emplace_back(candidateId, meta_data_predicates[candidateId].second);

                    // To Do insert Count min sketch Logic
                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                }

                // Two hop insertion
                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)this->get_linklist0(candidateId);
                    if (!twoHopData)
                        continue; // Error handling

                    size_t twoHopSize = this->getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        if (localVisitedIds.find(candidateIdTwoHop) != localVisitedIds.end())
                            continue;
                        localVisitedIds.insert(candidateIdTwoHop);
                        visitedIds.insert(candidateIdTwoHop);
                        // Insert CDF value here
                        // Update CMS
                        for (const auto &predicate : meta_data_predicates[candidateIdTwoHop].first)
                        {
                            current_cms.update(predicate, candidateIdTwoHop, 1);
                        }

                        predicate_data_CDF.emplace_back(candidateIdTwoHop, meta_data_predicates[candidateIdTwoHop].second);
                        // Todo the CMS value

                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }
                // Updating the data structure after insertion
                // Updating the data structure after insertion
                if (counterForFilter >= size_of_cluster)
                {
                    // Update CMS
                    current_cms.total = counterForFilter;
                    mapForCMS[clusterNumber] = std::move(current_cms);
                    current_cms = CountMinSketchMinHash(); // Create new CMS
                    // Update the CDF Regression Model
                    //  Step 1: Calculate the total number of data points
                    size_t n = predicate_data_CDF.size();
                    // Step 2: Count occurrences of each unique value// this computation is only for CDF computation
                    std::map<int, std::vector<tableint>> count_map;
                    for (const auto &pred : predicate_data_CDF)
                    {
                        // Check if the predicate (pred.second) already exists in the map
                        if (count_map.find(pred.second) != count_map.end())
                        {
                            // If it exists, add the id (pred.first) to the map vector of ids
                            count_map[pred.second].push_back(pred.first);
                        }
                        else
                        {
                            // If it doesn't exist, create a new vector with the id and insert it into the map
                            count_map[pred.second] = {pred.first};
                        }
                    }

                    // Step 3: Prepare a vector to store the CDF
                    std::vector<std::pair<int, double>> cdf; // (value, cumulative probability)
                    // Step 4: Compute the CDF
                    double cumulative_count = 0;
                    //  here count is the of ids vector
                    std::unordered_map<int, std::vector<uint16_t>> map_cdf_range_k_minwise;
                    for (const auto &[value, ids] : count_map)
                    {
                        cumulative_count += ids.size(); // Increment cumulative count

                        double cumulative_probability = cumulative_count / n; // Calculate cumulative probability
                        int label = check_range_search(cumulative_probability);

                        computing_and_inserting_relevant_range_to_vector(label, ids, map_cdf_range_k_minwise);

                        cdf.push_back({value, cumulative_probability});
                    }

                    RegressionModel reg_model;

                    reg_model.train_int(cdf);

                    reg_model.setMapCdfRangeKMinwiseFull(map_cdf_range_k_minwise);
                    reg_model.setTotal(counterForFilter);

                    mapForRegressionModel[clusterNumber] = std::move(reg_model);

                    clusterNumber++;
                    counterForFilter = 0;
                    predicate_data_CDF.clear();
                    cdf.clear();
                    localVisitedIds.clear();
                }
            }

            if (counterForFilter > 0)
            {
                // Update CMS
                current_cms.total = counterForFilter;
                mapForCMS[clusterNumber] = std::move(current_cms);
                // Update the CDF Regression Model

                // Step 1: Calculate the total number of data points
                size_t n = predicate_data_CDF.size();

                // Step 2: Count occurrences of each unique value// this computation is only for CDF computation
                std::map<int, std::vector<tableint>> count_map;
                for (const auto &pred : predicate_data_CDF)
                {
                    // Check if the predicate (pred.second) already exists in the map
                    if (count_map.find(pred.second) != count_map.end())
                    {
                        // If it exists, add the id (pred.first) to the vector
                        count_map[pred.second].push_back(pred.first);
                    }
                    else
                    {
                        // If it doesn't exist, create a new vector with the id and insert it into the map
                        count_map[pred.second] = {pred.first};
                    }
                }

                // Step 3: Prepare a vector to store the CDF
                std::vector<std::pair<int, double>> cdf; // (value, cumulative probability)

                // Step 4: Compute the CDF
                double cumulative_count = 0;
                //  here count is the size of ids vector
                std::unordered_map<int, std::vector<uint16_t>> map_cdf_range_k_minwise;
                for (const auto &[value, count] : count_map)
                {
                    cumulative_count += count.size(); // Increment cumulative count

                    double cumulative_probability = cumulative_count / n; // Calculate cumulative probability
                    int label = check_range_search(cumulative_probability);

                    computing_and_inserting_relevant_range_to_vector(label, count, map_cdf_range_k_minwise);

                    cdf.push_back({value, cumulative_probability});
                }
                /// Compute the Regression Model using Eign
                // Insert it into the  map insert it into the map[id, Model]
                //  Flush the predicate Vector
                // Also in the map also insert the CMS here.

                RegressionModel reg_model;
                reg_model.train_int(cdf);
                reg_model.setMapCdfRangeKMinwiseFull(map_cdf_range_k_minwise);
                reg_model.setTotal(counterForFilter);
                mapForRegressionModel[clusterNumber] = std::move(reg_model);
                clusterNumber++;
                counterForFilter = 0;
                predicate_data_CDF.clear();
                cdf.clear();
                localVisitedIds.clear();
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

        template <typename T>
        void computing_and_inserting_relevant_range_to_vector(
            int &range_no,
            const std::vector<tableint> &ids,
            std::unordered_map<int, std::vector<T>> &map_cdf_range_k_minwise)
        {
            // Safety checks
            static_assert(std::is_unsigned<T>::value, "T must be unsigned");
            static_assert(sizeof(T) == 1 || sizeof(T) == 2, "T must be uint8_t or uint16_t");

            // Get or create the vector for this range
            std::vector<T> &fingerprint_vec = map_cdf_range_k_minwise[range_no];

            // -------------------------------
            // Bottom-k selection using FULL 64-bit hashes
            // -------------------------------
            std::priority_queue<uint64_t> heap; // max-heap

            for (auto id : ids)
            {
                uint64_t h = MurmurHash64B(&id, sizeof(id), SEED);

                if (heap.size() < SIZE_Of_K_Min_WISE_HASH)
                {
                    heap.push(h);
                }
                else if (h < heap.top())
                {
                    heap.pop();
                    heap.push(h);
                }
            }

            // Clear old vector (optional, if rebuilding)
            fingerprint_vec.clear();

            // -------------------------------
            // Truncate AFTER bottom-k selection
            // -------------------------------
            while (!heap.empty())
            {
                uint64_t h = heap.top();
                heap.pop();

                // Use MSB bits for T-bit fingerprint
                T fingerprint = static_cast<T>(h >> (64 - sizeof(T) * 8));
                fingerprint_vec.push_back(fingerprint);
            }

            // -------------------------------
            // Sort vector for SIMD intersection
            // -------------------------------
            std::sort(fingerprint_vec.begin(), fingerprint_vec.end());

            // If vector exceeds SIZE_Of_K_Min_WISE_HASH, trim largest
            if (fingerprint_vec.size() > SIZE_Of_K_Min_WISE_HASH)
            {
                fingerprint_vec.resize(SIZE_Of_K_Min_WISE_HASH);
            }
        }

        // Search mechanisim
        void predicateCondition(char *filters_array)
        {
            filter_id_map = filters_array;
        }

        // Searching
        std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement> prediction(const void *query_data, size_t query_number, const std::pair<int, int> &query_range_predicates, const std::vector<std::string> &query_point_predicates, size_t top_k)
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
            const double popularity_threshold = 0.20; // Tuneabable threshold for popularity

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

            for (tableint id : indices)
            {
                auto clusters = cluster_contains_attribute(id);
                auto [cluster_id, max_pop] = popularity_computation(clusters, query_point_predicates, query_range_predicates);
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

        // Popularity computation function for predictive point queries
        std::pair<size_t, double>
        popularity_computation(
            std::vector<unsigned int> &clusters,
            const std::vector<std::string> &query_predicates_point,
            const std::pair<int, int> &query_predicates_range)
        {
            int max_intersection = 0;
            size_t cluster_id = 0;
            size_t total_count = 0;

            for (const auto &cid : clusters)
            {
                auto cms_it = mapForCMS.find(cid);
                if (cms_it == mapForCMS.end())
                    continue;

                auto reg_it = mapForRegressionModel.find(cid);
                if (reg_it == mapForRegressionModel.end())
                    continue;

                auto &reg_model = reg_it->second;
                CountMinSketchMinHash &cms = cms_it->second;

                auto estimate_result = cms.estimate(query_predicates_point[0]);
                auto &CMSkMinWiseVector =
                    cms.full_keys[estimate_result.first][estimate_result.second];

                int left_label = check_range_search(
                    reg_model.predict_int(query_predicates_range.first));
                int right_label = check_range_search(
                    reg_model.predict_int(query_predicates_range.second));

                if (left_label > right_label)
                    std::swap(left_label, right_label);

                auto &ranges_ = reg_model.getMapCdfRangeKMinwiseFull();

                int num_threads = omp_get_max_threads();
                std::vector<std::vector<uint16_t>> local_vectors(num_threads);

#pragma omp parallel
                {
                    int tid = omp_get_thread_num();

#pragma omp for schedule(static) nowait
                    for (int i = left_label; i <= right_label; i++)
                    {
                        auto range_it = ranges_.find(i);
                        if (range_it == ranges_.end())
                            continue;

                        for (uint16_t elem : range_it->second)
                        {
                            if (CMSkMinWiseVector.count(elem))
                                local_vectors[tid].push_back(elem);
                        }
                    }
                }

                std::unordered_set<uint16_t> seen;
                for (const auto &vec : local_vectors)
                    seen.insert(vec.begin(), vec.end());

                if (static_cast<int>(seen.size()) > max_intersection)
                {
                    max_intersection = static_cast<int>(seen.size());
                    cluster_id = cid;
                    total_count = std::min(CMSkMinWiseVector.size(),
                                           static_cast<size_t>(cms.keys));
                }
            }

            double percentage_intersection =
                total_count > 0
                    ? static_cast<double>(max_intersection) / total_count
                    : 0.0;

            return {cluster_id, percentage_intersection};
        }

        // Cluster containing the id
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
    };
}