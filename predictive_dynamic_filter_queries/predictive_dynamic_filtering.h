#pragma once
#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"
#include "../predictive_range_queries/ranges.h"
#include "../predictive_point_queries/memory_access.h"
#include "../predictive_point_queries/count_min_sketch_min_hash.h"
#include "../predictive_range_queries/regression.h"
#include "hnswlib/visited_list_pool.h"
#include <vector>
#include <cstring>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>
#include <map>
#include <omp.h>
namespace clustered_hybrid_search
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    template <typename dist_t>
    class PredictiveDynamicFilteringHNSW : public hnswlib::HierarchicalNSW<dist_t>
    {
        // Comparator
        struct CompareByFirstElement
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };
        using Candidate = std::pair<dist_t, tableint>;
        using CandidateQueue = std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement>;

    public:
        std::unordered_map<tableint, std::vector<std::string>> meta_data_predicates;
        std::unordered_map<unsigned int, CountMinSketchMinHash> mapForCMS;
        std::unordered_map<unsigned int, RegressionModel> mapForRegressionModel;
        std::unordered_map<tableint, std::unordered_set<unsigned int>> cluster_Mem_chk;
        std::unique_ptr<hnswlib::VisitedListPool> visited_list_pool_{nullptr};

        char *mem_for_ids_for_clusters{nullptr};
        char *filter_id_map;
        size_t max_elements_;
        float popularity_threshold;
        std::unordered_map<size_t, std::pair<float, float>> cms_bounds_cache; // Cache for CMS bounds

    public:
        PredictiveDynamicFilteringHNSW(hnswlib::SpaceInterface<dist_t> *space, size_t max_elements, const std::string &location_of_index, std::unordered_map<tableint, std::vector<std::string>> &meta_data_predicates_)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements)
        {

            this->loadIndex(location_of_index, space, max_elements);
            meta_data_predicates = meta_data_predicates_;
            mem_for_ids_for_clusters = (char *)malloc(max_elements * (3 * sizeof(char)));
            memset(mem_for_ids_for_clusters, 0, max_elements * (3 * sizeof(char)));
            max_elements_ = max_elements;
            visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
            popularity_threshold = 0.0f;
            // Open the file in binary mode
            if (mem_for_ids_for_clusters == nullptr)
                throw std::runtime_error("Not enough memory");
        }

        PredictiveDynamicFilteringHNSW(hnswlib::SpaceInterface<dist_t> *space, size_t max_elements, const std::string &location_of_index)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements)
        {

            this->loadIndex(location_of_index, space, max_elements);
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

                        // current_cms.total = current_cms.total + 1; // Increment total count for C
                    }

                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                    // current_cms.total = counterForFilter; // Increment total count for CMS
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

                            // current_cms.total = current_cms.total + 1;
                        }

                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);
                        counterForFilter++;
                    }
                }

                // Update data structure when cluster is full
                // if (counterForFilter >= size_of_cluster)
                // {
                // ✅ Use move semantics

                mapForCMS[clusterNumber] = std::move(current_cms);
                // std::cout << "Cluster " << clusterNumber << " has " << counterForFilter << " items." << current_cms.totalcount() << "TestCounter" << test_counter << std::endl;
                // Reset for next cluster
                clusterNumber++;
                counterForFilter = 0;

                current_cms = CountMinSketchMinHash(); // Create new CMS
                visitedIdsLocally.clear();             // Clear local visited IDs
                // }
            }

            // Handle remaining nodes
            // if (counterForFilter > 0)
            // {

            //     mapForCMS[clusterNumber] = std::move(current_cms);
            // }
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

        void popularityThresoldComputation()
        {

            // 1. Get CMS ratio bounds
            auto bounds = lowerAndUpperBoundForCms();
            float lowerBound = bounds.first;
            float upperBound = bounds.second;

            // 2. Sanity check
            if (upperBound <= lowerBound)
                return;

            // 3. Clamp efs to maximum
            const float MAX_EFS = 1300.0f;
            float efsClamped = std::min(static_cast<float>(this->ef_), MAX_EFS);

            // 4. Normalize efs using log scaling (0 → 1)
            float alpha =
                std::log1p(efsClamped) /
                std::log1p(MAX_EFS);

            // 5. Compute popularity threshold
            popularity_threshold =
                lowerBound + alpha * (upperBound - lowerBound);
            std::cout<<popularity_threshold<<std::endl;
        }
        // Search Strategies

        void searchPointQueries(const void *query_data, size_t query_number, size_t start, size_t k, std::vector<std::string> &query_attribute)

        {

          
            Candidate result;

            tableint currObj = this->enterpoint_node_;
            dist_t curdist = this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);
            size_t filter_offset = (query_number - start) * max_elements_;

            for (int level = this->maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)this->get_linklist(currObj, level);
                    int size = this->getListCount(data);
                    this->metric_hops++;
                    this->metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        // if (filter_id_map[filter_offset + cand])
                        // {
                        dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(cand), this->dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                        //}
                        else
                        {
                            continue;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirstElement> top_candidates =
                searchBaseLayerST(
                    currObj, query_data, std::max(this->ef_, k), query_number, start, query_attribute);

            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }
            std::vector<std::pair<dist_t, size_t>> results;
            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                results.push_back(std::pair<dist_t, size_t>(rez.first, this->getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            std::string filename = "/scratch/aa5f25/datasets/yt8m/results/" + std::to_string(this->ef_);
            create_directory_if_not_exists(filename);
            filename += "/Q" +
                        std::to_string(query_number) + ".csv";
            std::ofstream out(filename);
            if (!out.is_open())
            {
                std::cerr << "❌ Failed to open file: " << filename << std::endl;
                return;
            }

            out << "ID,Distance\n";
            if (!results.empty())
            {
                for (const auto &p : results)
                    out << p.second << "," << p.first << "\n";
            }
            else
            {
                // 🔹 Fallback: no results → write k rows of dummy values
                for (size_t i = 0; i < k; i++)
                    out << -1 << "," << std::numeric_limits<float>::max() << "\n";
            }

            out.close();
            // std::cout << "✅ Saved results for query " << query_number << " to " << filename << std::endl;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirstElement>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, size_t query_number, size_t start, std::vector<std::string> &query_attribute)
        {
            const size_t filter_offset = (query_number - start) * max_elements_;
            //  double threshold = 0.2f;
            std::unordered_set<tableint> cluster_visited;
            hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirstElement> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirstElement> candidate_set;
            dist_t lowerBound;
            char *ep_data = this->getDataByInternalId(ep_id);
            dist_t dist = this->fstdistfunc_(data_point, ep_data, this->dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t candidate_dist = -current_node_pair.first;
                tableint current_node_id = current_node_pair.second;
                //   std::cout << "Candidate Set Entry" << candidate_set.size() << std::endl;
                int *data = (int *)this->get_linklist0(current_node_id);
                candidate_set.pop();
                auto clusters = cluster_contains_attribute(current_node_id);
                auto [cluster_id, max_pop] = popularity_computation(clusters, query_attribute);
                // Decide which search to perform
                if (max_pop >popularity_threshold) // High popularity → twoHopSearch    if (max_pop > popularity_threshold)
                {

                    if (cluster_visited.find(cluster_id) == cluster_visited.end())
                    {

                        // First visit to this cluster → twoHopSearch
                        oneHopSearch(data_point, current_node_id, filter_offset,
                                     top_candidates, candidate_set, lowerBound, ef,
                                     visited_array, visited_array_tag);
                    }
                    else
                    {
                        // Cluster already visited → fallback to oneHopSearch
                        oneHopSearch(data_point, current_node_id, filter_offset,
                                     top_candidates, candidate_set, lowerBound, ef,
                                     visited_array, visited_array_tag);
                    }
                }
                else
                {
                    // Low popularity → oneHopSearch

                    twoHopSearch(data_point, current_node_id, filter_offset,
                                 top_candidates, candidate_set, lowerBound, ef,
                                 visited_array, visited_array_tag);
                }

                // Mark cluster as visited
                cluster_visited.insert(cluster_id);
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }
        // Cluster if already visted what is its frequency:

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
        std::pair<size_t, double> popularity_computation(std::vector<unsigned int> &clusters, const std::vector<std::string> &query_predicates)
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

        void twoHopSearch(
            const void *query_data,
            tableint node,
            const size_t &filter_offset,
            CandidateQueue &top_candidates,
            CandidateQueue &candidate_set,
            dist_t &lowerBound,
            const size_t &ef,
            hnswlib::vl_type *visited_array,
            hnswlib::vl_type visited_array_tag = 0)

        {
            int *data;
            data = (int *)this->get_linklist0(node);
            size_t size = this->getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

//             // Getting the nearest neighbours size
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);

#endif

            for (size_t j = 0; j < size; j++)
            {
                tableint candidate_id = *(datal + j);

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
#endif

                // bool is_visited = visited_bitmap[byte_index_cand] & (1 << bit_index_cand);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;

                // visited_bitmap[byte_index_cand] |= (1 << bit_index_cand);

                if (filter_id_map[filter_offset + candidate_id]) // Check if the item satisfy the predicate

                {
                    // std::cout<<"I am here---Searching"<<candidate_id<<std::endl;

                    char *currObj1 = (this->getDataByInternalId(candidate_id));
                    visited_array[candidate_id] = visited_array_tag;
                    // dist_t dist = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);
                    dist_t dist = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);
                    updatePriorityQueue(candidate_id, dist, top_candidates, candidate_set, lowerBound, ef);
                }
                // Two hop searching
                int *twoHopData = (int *)this->get_linklist0(candidate_id);
                if (!twoHopData)
                    continue; // Error handling

                size_t twoHopSize = this->getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    // size_t byte_index_two_hop_cand = candidateIdTwoHop / 8;
                    // size_t bit_index_two_hop_cand = candidateIdTwoHop % 8;
                    // bool is_visited = visited_bitmap[byte_index_two_hop_cand] & (1 << bit_index_two_hop_cand);
                    if (visited_array[candidateIdTwoHop] == visited_array_tag)
                        continue;

                    // visited_bitmap[byte_index_two_hop_cand] |= (1 << bit_index_two_hop_cand);

                    if (filter_id_map[filter_offset + candidateIdTwoHop]) // Check if the item satisfy the predicate
                    {

                        char *currObj1 = (this->getDataByInternalId(candidateIdTwoHop));

                        // dist_t dist1 = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);

                        dist_t dist1 = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);
                        visited_array[candidateIdTwoHop] = visited_array_tag;
                        updatePriorityQueue(candidateIdTwoHop, dist1, top_candidates, candidate_set, lowerBound, ef);

                        //  result_vector.push_back(std::make_pair(candidateIdTwoHop, dist1)); call methods
                    }
                }
            }

            // visited_list_pool_->releaseVisitedList(vl);
        }
        void oneHopSearch(
            const void *query_data,
            tableint node,
            const size_t &filter_offset,
            CandidateQueue &top_candidates,
            CandidateQueue &candidate_set,
            dist_t &lowerBound,
            const size_t &ef,
            hnswlib::vl_type *visited_array,
            hnswlib::vl_type visited_array_tag = 0)

        {
            int *data;
            data = (int *)this->get_linklist0(node);
            size_t size = this->getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

//             // Getting the nearest neighbours size
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);

#endif

            for (size_t j = 0; j < size; j++)
            {
                tableint candidate_id = *(datal + j);

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
#endif

                // bool is_visited = visited_bitmap[byte_index_cand] & (1 << bit_index_cand);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;

                // visited_bitmap[byte_index_cand] |= (1 << bit_index_cand);

                if (filter_id_map[filter_offset + candidate_id]) // Check if the item satisfy the predicate

                {
                    char *currObj1 = (this->getDataByInternalId(candidate_id));
                    dist_t dist = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);
                    visited_array[candidate_id] = visited_array_tag;
                    updatePriorityQueue(candidate_id, dist, top_candidates, candidate_set, lowerBound, ef);
                }
                // Two hop searching
            }

            // visited_list_pool_->releaseVisitedList(vl);
        }

        /**
         * @brief Updates the candidate and top priority queues with a new node.
         *
         * @param candidate_id     ID of the candidate node.
         * @param dist             Distance of the candidate to the query.
         * @param top_candidates   Priority queue of current best candidates.
         * @param candidate_set    Priority queue of nodes to explore.
         * @param lowerBound       Current search bound, updated if needed.
         * @param ef               Maximum size of the top_candidates queue.
         */

        void updatePriorityQueue(size_t candidate_id, dist_t dist, CandidateQueue &top_candidates, CandidateQueue &candidate_set, dist_t &lowerBound, size_t ef)
        {
            // Decide whether to consider candidate

            bool flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;

            if (!flag_consider_candidate)
                return;

            // Always insert into candidate set
            candidate_set.emplace(-dist, candidate_id);

#ifdef USE_SSE
            _mm_prefetch(this->data_level0_memory_ + candidate_set.top().second * this->size_data_per_element_ +
                             this->offsetLevel0_,
                         _MM_HINT_T0);
#endif

            // Always insert into top_candidates since bare_bone_search = true
            top_candidates.emplace(dist, candidate_id);

            // Remove extra if needed
            while (top_candidates.size() > ef)
            {
                //  auto removed = top_candidates.top(); // pair<dist, id>
                top_candidates.pop();
            }

            if (!top_candidates.empty())
                lowerBound = top_candidates.top().first;
        }
        // Itereate all count min sketches to find the lower and upper bound.
        std::pair<float, float> lowerAndUpperBoundForCms()
        {
            float globalMin = std::numeric_limits<float>::max();
            float globalMax = std::numeric_limits<float>::lowest();
            bool foundAny = false;

            for (auto &it : mapForCMS)
            {
                CountMinSketchMinHash &cms = it.second;

                if (cms.totalcount() == 0)
                    continue;

                //  std::cout<<"Total Testing"<<cms.totalcount()<<std::endl;

                float minRatio = std::numeric_limits<float>::max();
                float maxRatio = std::numeric_limits<float>::lowest();
                bool foundNonZero = false;

                for (unsigned int i = 0; i < cms.d; ++i)
                {
                    for (unsigned int j = 0; j < cms.w; ++j)
                    {
                        int count = cms.C[i][j];
                        if (count == 0)
                            continue;

                        float ratio =
                            static_cast<float>(count) /
                            static_cast<float>(cms.totalcount());

                        minRatio = std::min(minRatio, ratio);
                        maxRatio = std::max(maxRatio, ratio);
                        foundNonZero = true;
                    }
                }

                if (!foundNonZero)
                    continue;

                globalMin = std::min(globalMin, minRatio);
                globalMax = std::max(globalMax, maxRatio);
                foundAny = true;
            }

            // no valid data at all
            if (!foundAny)
                return {0.0f, 0.0f};

            return {globalMin, globalMax};
        }

        void freeMemory()
        {
            if (mem_for_ids_for_clusters)
            {
                free(mem_for_ids_for_clusters);
                mem_for_ids_for_clusters = nullptr;
            }
            for (auto &pair : mapForCMS)
            {
                CountMinSketchMinHash &cms = pair.second;
                // Delete each row (inner array)
                for (unsigned int i = 0; i < cms.d; ++i)
                {
                    delete[] cms.C[i];
                }
                // Delete the outer array
                delete[] cms.C;
                cms.C = nullptr; // Avoid dangling pointer

                for (unsigned int i = 0; i < cms.d; ++i)
                {
                    delete[] cms.hashes[i];
                }
                // Delete the outer array
                delete[] cms.hashes;
                cms.hashes = nullptr; // Avoid dangling pointer
            }
            mapForCMS.clear(); // Clear the map after destruction
        }

        void groundTruthBatch(const float *query_vec,
                              size_t k,
                              size_t query_id_global,
                              size_t total_elements,
                              size_t batch_start)
        {
            if (query_id_global < batch_start)
            {
                std::cerr << "❌ Invalid query index\n";
                return;
            }

            const size_t local_index = query_id_global - batch_start;
            const size_t filter_offset = local_index * total_elements;

            std::priority_queue<std::pair<dist_t, size_t>> top_candidates;

            for (size_t i = 0; i < total_elements; ++i)
            {
                if (!filter_id_map[filter_offset + i])
                    continue;

                const char *ep_data = this->getDataByInternalId(i);
               // std::cout<<"I ma here"<<std::endl;
                dist_t dist = this->fstdistfunc_(
                    query_vec,
                    ep_data,
                    this->dist_func_param_);

                top_candidates.emplace(dist, i);

                if (top_candidates.size() > k)
                    top_candidates.pop();
            }

            std::vector<std::pair<dist_t, size_t>> results;
            results.reserve(k);

            while (!top_candidates.empty())
            {
                results.push_back(top_candidates.top());
                top_candidates.pop();
            }
            std::reverse(results.begin(), results.end());

            std::string filename =
                "/scratch/aa5f25/datasets/yt8m/Ground_truth/Q" +
                std::to_string(query_id_global) + ".csv";

            std::ofstream out(filename);
            out << "ID,Distance\n";

            if (!results.empty())
            {
                for (auto &p : results)
                    out << p.second << "," << p.first << "\n";
            }
            else
            {
                for (size_t i = 0; i < k; ++i)
                    out << -1 << "," << std::numeric_limits<dist_t>::max() << "\n";
            }
        }

        void create_directory_if_not_exists(const std::string &path)
        {
            if (mkdir(path.c_str(), 0777) == -1)
            {
                if (errno == EEXIST)
                {
                    // Directory already exists, that's fine
                }
                else
                {
                    std::cerr << "Error creating directory: " << path << std::endl;
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirstElement>
        searchBaseLayerST_Base(tableint ep_id, const void *data_point, size_t ef, size_t query_number, size_t start, std::vector<std::string> &query_attribute)
        {
            const size_t filter_offset = (query_number - start) * max_elements_;
            hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirstElement> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirstElement> candidate_set;

            dist_t lowerBound;

            char *ep_data = this->getDataByInternalId(ep_id);
            dist_t dist = this->fstdistfunc_(data_point, ep_data, this->dist_func_param_);

            lowerBound = dist;
            if (filter_id_map[filter_offset + ep_id])
            {
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t candidate_dist = -current_node_pair.first;

                bool flag_stop_search;

                flag_stop_search = candidate_dist > lowerBound;

                if (flag_stop_search)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)this->get_linklist0(current_node_id);
                size_t size = this->getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(this->data_level0_memory_ + (*(data + 1)) * this->size_data_per_element_ + this->offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(this->data_level0_memory_ + (*(data + j + 1)) * this->size_data_per_element_ + this->offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (this->getDataByInternalId(candidate_id));
                        dist_t dist = this->fstdistfunc_(data_point, currObj1, this->dist_func_param_);

                        bool flag_consider_candidate;

                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;

                        if (flag_consider_candidate)
                        {
                            if (filter_id_map[filter_offset + candidate_id]) // Check if the item satisfy the predicate
                            {
                                candidate_set.emplace(-dist, candidate_id);

#ifdef USE_SSE
                                _mm_prefetch(this->data_level0_memory_ + candidate_set.top().second * this->size_data_per_element_ +
                                                 this->offsetLevel0_, ///////////
                                             _MM_HINT_T0);            ////////////////////////
#endif

                                top_candidates.emplace(dist, candidate_id);
                            }

                            bool flag_remove_extra = false;

                            flag_remove_extra = top_candidates.size() > ef;

                            while (flag_remove_extra)
                            {

                                top_candidates.pop();

                                flag_remove_extra = top_candidates.size() > ef;
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        
    };
}
