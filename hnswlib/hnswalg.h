#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "memoryaccess.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "bloom.h"
#include <chrono>
#include <dirent.h>
#include <mutex>
#include <future>
#include <map>
#include <fstream> // Include this at the top
#include <sys/stat.h>

#include "Node.h"
#include "linktree.h"
#include "regression.h"
#include "Ranges.h"
#include "count_min_sketch_min_hash.hpp"

#include <thread>
#include <mutex>

namespace hnswlib
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template <typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t>
    {
    public:
        static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
        static const unsigned char DELETE_MARK = 0x01;

        size_t max_elements_{0};
        mutable std::atomic<size_t> cur_element_count{0}; // current number of elements
        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        mutable std::atomic<size_t> num_deleted_{0}; // number of deleted elements
        size_t M_{0};
        size_t maxM_{0};
        size_t maxM0_{0};
        size_t ef_construction_{0};
        size_t ef_{0};

        double mult_{0.0}, revSize_{0.0};
        int maxlevel_{0};

        std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

        // Locks operations with element by label value
        mutable std::vector<std::mutex> label_op_locks_;

        std::mutex global;
        std::vector<std::mutex> link_list_locks_;

        tableint enterpoint_node_{0};

        size_t size_links_level0_{0};
        size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

        char *data_level0_memory_{nullptr};
        char **linkLists_{nullptr};
        std::vector<int> element_levels_; // keeps level of each element

        size_t data_size_{0};
        std::mutex result_mutex;

        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};

        mutable std::mutex label_lookup_lock; // lock for label_lookup_
        std::unordered_map<labeltype, tableint> label_lookup_;

        //  Metadata map
        std::unordered_map<tableint, std::vector<char *>> meta_data_predicates;
        /// Count-Min Sketch and Bloom Filter
        std::unordered_map<unsigned int, CountMinSketchMinHash> mapForCMS;
        // Map for BloomFilter
        std::unordered_map<int, BloomFilter> mapForBF;

        std::unordered_map<size_t, std::unordered_map<std::string, unsigned int>> correctMap;

        // Map for cluster Optional:
        std::unordered_map<tableint, std::unordered_set<unsigned int>> cluster_hash_based;
        // Mutex
        std::mutex popularity_mutex;

        // Just insertion for coorelation
        std::unordered_map<tableint, unsigned int> corelation_id;
        // Count _min width
        int count_min_width{1000};
        int count_min_height{3};
        int bloom_filter_size{10000};
        // Result counter
        int result_counter_distance_computation{0};
        int result_counter{0};
        int result_counter_two_hop{0};
        int result_counter_one_hop{0};
        // Counting Quotient Filter

        uint64_t qbits = 20;
        uint64_t rbits = 4;
        uint64_t nhashbits = qbits + rbits;
        uint64_t nslots = (1ULL << qbits);
        uint64_t nvals = 95 * nslots / 100;
        uint64_t key_count = 1;
        // uint64_t *vals;
        unsigned int *clusterIDs;
        std::unordered_map<tableint, std::unordered_set<unsigned int>> cluster_Mem_chk;
        char *mem_for_ids_clusters{nullptr};

        std::vector<std::pair<int, std::set<char> *>> set_intersection = {};

        std::unordered_map<int, int> popularity_map;

        // Optimal memory management methodology

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        mutable std::atomic<long> metric_distance_computations{0};
        mutable std::atomic<long> metric_hops{0};

        bool allow_replace_deleted_ = false; // flag to replace deleted elements (marked as deleted) during insertions

        std::mutex deleted_elements_lock;              // lock for deleted_elements
        std::unordered_set<tableint> deleted_elements; // contains internal ids of deleted elements

        std::map<unsigned int, Vertex<std::string> *> searchMap;
        // Map for linear vector
        std::unordered_map<unsigned int, RegressionModel> mapForRegressionModel;
        // Map for range search
        std::unordered_map<unsigned int, std::pair<CountMinSketchMinHash, RegressionModel>> map_for_hybrid_range_queries;

        int count = 0;
        // The clustering approach for finding the key
        std::unordered_map<unsigned int, BLinkTree<std::string, unsigned int> *> cluster_and_associated_tree;
        // Meta_Data_Information:
        std::unordered_map<tableint, std::pair<std::string, std::vector<char *>>> meta_data_multidiemesional_query;
        // Un ordered map for key key
        std::unordered_map<int, std::pair<double, int>> predicted_CDF_map;

        std::unordered_map<unsigned int, unsigned int> meta_data_int_;
        // Memory Declared for disk optimization
        char *bit_array_for_disk_access; // = (char *)malloc(array_size_in_bytes);
        mutable int counter_or_disk_access;
        // Compute the

        HierarchicalNSW(SpaceInterface<dist_t> *s)
        {
        }

        // Some Items for initialization
        HierarchicalNSW(
            SpaceInterface<dist_t> *s,
            const std::string &location,
            std::unordered_map<tableint, std::vector<char *>> &meta_data_predicates_,
            int total_elememt,
            bool nmslib = false,
            size_t max_elements = 0,
            bool allow_replace_deleted = false)
            : allow_replace_deleted_(allow_replace_deleted)
        {
            loadIndex(location, s, max_elements);
            // Initilization
            max_elements_ = total_elememt;
            meta_data_predicates = meta_data_predicates_;
            clusterIDs = new unsigned int[max_elements_]();
            // Initializing it when there is no file their.
            counter_or_disk_access = 0;
            mem_for_ids_clusters = (char *)malloc(max_elements_ * (3 * sizeof(char)));
            memset(mem_for_ids_clusters, 0, max_elements_ * (3 * sizeof(char)));
            // Open the file in binary mode
            if (mem_for_ids_clusters == nullptr)
                throw std::runtime_error("Not enough memory");
        }

        // Some Items for initialization
        HierarchicalNSW(
            SpaceInterface<dist_t> *s,
            const std::string &location,
            std::unordered_map<tableint, std::vector<char *>> &meta_data_predicates_,

            std::unordered_map<unsigned int, std::pair<std::string, std::vector<char *>>> &multi_diemesional_meta_data,
            std::unordered_map<unsigned int, unsigned int> &meta_data_int,
            int total_elememt,
            std::unordered_map<unsigned int, CountMinSketchMinHash> &mapForCMS_,
            bool cluster_writing_reading,
            std::map<unsigned int, Vertex<std::string> *> &rangeSearchMap,
            bool nmslib = false,
            size_t max_elements = 0,
            bool allow_replace_deleted = false)
            : allow_replace_deleted_(allow_replace_deleted)
        {
            loadIndex(location, s, max_elements);
            // Initilization
            max_elements_ = total_elememt;
            meta_data_predicates = meta_data_predicates_;

            meta_data_multidiemesional_query = multi_diemesional_meta_data;
            meta_data_int_ = meta_data_int;
            mapForCMS = mapForCMS_; // Map initization
            clusterIDs = new unsigned int[max_elements_]();
            // Initializing it when there is no file their.
            searchMap = rangeSearchMap;
            counter_or_disk_access = 0;

            mem_for_ids_clusters = (char *)malloc(max_elements_ * (3 * sizeof(char)));
            memset(mem_for_ids_clusters, 0, max_elements_ * (3 * sizeof(char)));
            // Open the file in binary mode
            if (cluster_writing_reading)
                readIdsAndClusterRelationShip("/data4/hnsw/paper/Clusters/map_16_32.bin", "/data4/hnsw/paper/Clusters/short_16_32.bin");

            if (mem_for_ids_clusters == nullptr)
                throw std::runtime_error("Not enough memory");
        }

        HierarchicalNSW(
            SpaceInterface<dist_t> *s,
            size_t max_elements,
            size_t M = 16,
            size_t ef_construction = 200,
            size_t random_seed = 100,
            bool allow_replace_deleted = false)
            : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
              link_list_locks_(max_elements),
              element_levels_(max_elements),
              allow_replace_deleted_(allow_replace_deleted)
        {
            max_elements_ = max_elements;
            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            count = 0;
            if (M <= 10000)
            {
                M_ = M;
            }
            else
            {
                HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
                HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
                M_ = 10000;
            }
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction, M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

            // initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;

            /// Inititalization

            clusterIDs = new unsigned int[max_elements]();

            mem_for_ids_clusters = (char *)malloc(max_elements_ * (3 * sizeof(char)));
            memset(mem_for_ids_clusters, 0, max_elements_ * (3 * sizeof(char)));

            if (mem_for_ids_clusters == nullptr)
                throw std::runtime_error("Not enough memory");
        }

        ~HierarchicalNSW()
        {

            delete[] clusterIDs;

            clear();
        }

        void clear()
        {
            free(data_level0_memory_);
            data_level0_memory_ = nullptr;
            for (tableint i = 0; i < cur_element_count; i++)
            {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            linkLists_ = nullptr;
            cur_element_count = 0;
            visited_list_pool_.reset(nullptr);

            free(mem_for_ids_clusters);
            mem_for_ids_clusters = nullptr;
        }

        struct CompareByFirst
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };

        void setEf(size_t ef)
        {
            ef_ = ef;
        }

        inline std::mutex &getLabelOpMutex(labeltype label) const
        {
            // calculate hash
            size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);

            return label_op_locks_[lock_id];
        }

        inline labeltype getExternalLabel(tableint internal_id) const
        {
            labeltype return_label;
            memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const
        {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const
        {

            return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const
        {

            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size)
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int)r;
        }

        size_t getMaxElements()
        {
            return max_elements_;
        }

        size_t getCurrentElementCount()
        {
            return cur_element_count;
        }

        size_t getDeletedCount()
        {
            return num_deleted_;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer)
        {

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;

            if (!isMarkedDeleted(ep_id))
            {

                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }

            else
            {

                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            // std::exit(0);

            while (!candidateSet.empty())
            {
                // getting the level for candidate set
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_)
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second; // Entry point id curNode for first iteration

                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {

                    data = (int *)get_linklist0(curNodeNum); // layer id i think
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data); // What is the value of data is actual the size her

                tableint *datal = (tableint *)(data + 1);

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);

                    //                    if (candidate_id == 0) continue;

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
        template <bool bare_bone_search = true, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(
            tableint ep_id,
            const void *data_point,
            size_t ef,
            BaseFilterFunctor *isIdAllowed = nullptr, unsigned int &left_range = 0, unsigned int &right_range = 0, std::vector<char *> additionalData = std::vector<char *>(),
            BaseSearchStopCondition<dist_t> *stop_condition = nullptr) const
        {

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::unordered_set<tableint> visitedNodes;

            int testing_count = 0;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_ACORN1;
            // std::vector<std::pair<int, std::set<char> *>> set_intersec;

            int selected_cluster = -2;

            std::vector<std::string> string_vector;

            // Iterate over char_vector and convert each char* to std::string
            for (char *key : additionalData)
            {
                string_vector.push_back(std::string(key));
            }

            dist_t lowerBound;
            if (bare_bone_search ||
                (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))))
            {
                char *ep_data = getDataByInternalId(ep_id);
                dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);

                // ACORN
                // if (result_computer_check(additionalData, ep_id))
                // {

                //     //     //     //  if(range_int_computer(left_range, right_range, ep_id)){

                //     //     // // //     //     // //             //  unsigned int selected_cluster = -1;

                //     //     // // //     //     // //             // int popularity_result = compute_popularity(ep_id, additionalData, set_intersec, selected_cluster);
                //   top_candidates_ACORN1.emplace(dist, ep_id);
                //     //     // // //     //     // // //         //     // two_hop_search_ACORN(data_point, ep_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);
                //     //     // // //     //     // // //         //     //two_hop_search_ACORN_RANGE(data_point, ep_id, vl, visitedNodes, left_range, right_range, top_candidates_ACORN1);
                //     //     // // //     //     // // //    two_hop_search_ACORN(data_point, ep_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);
                //   }

                //     int popularity_result = compute_corelation_pop(ep_id, string_vector);

                // //   cout << " " << popularity_result << endl;
                // //     // int popularity_result = compute_pop_during_exhasutive(ep_id, additionalData);
                //     if (popularity_result > 1){
                // two_hop_search_ACORN(data_point, ep_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);
                //     }
                //     else{
                //          one_hop_search_ACORN(data_point, ep_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);

                //     }

                //  testing_count=testing_count+1;
                //  pair<double, int> CDF_Difference = predictedCDF(ep_id,left_range, right_range, ep_id);
                // if(CDF_Difference.first<0.2)
                //  two_hop_search_ACORN_RANGE(data_point, ep_id, vl, visitedNodes, left_range, right_range, top_candidates_ACORN1);
                // if (multidiemensional_search(ep_id, left_range, right_range, additionalData))
                // {
                //     top_candidates_ACORN1.emplace(dist, ep_id);
                // }
                // two_hop_search_multidiemension_ACORN(data_point, vl, additionalData, left_range, right_range, ep_id, visitedNodes, top_candidates_ACORN1);

                if (!bare_bone_search && stop_condition)
                {

                    stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
                }
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t candidate_dist = -current_node_pair.first;

                bool flag_stop_search;
                if (bare_bone_search)
                {

                    flag_stop_search = candidate_dist > lowerBound;
                }
                else
                {
                    if (stop_condition)
                    {
                        flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                    }
                    else
                    {
                        flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                    }
                }
                if (flag_stop_search)
                {
                    break;
                }

                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);

                if (collect_metrics)
                {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
                    //                    if (candidate_id == 0) continue;
                    // if (!result_computer_check(additionalData, candidate_id))
                    //     continue;

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        bool flag_consider_candidate;
                        if (!bare_bone_search && stop_condition)
                        {
                            flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                        }
                        else
                        {
                            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                        }

                        if (flag_consider_candidate)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if (bare_bone_search ||
                                (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id)))))
                            {
                                top_candidates.emplace(dist, candidate_id);
                                testing_count = testing_count + 1;
                                // ACORN Search a loop to search

                                // if (range_result_computer_check(left_range, right_range, candidate_id))
                                // {
                                //     top_candidates_ACORN1.emplace(dist, candidate_id);
                                //     // two_hop_search_ACORN(data_point, ep_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);
                                //    // two_hop_search_ACORN_RANGE(data_point, candidate_id, vl, visitedNodes, left_range, right_range, top_candidates_ACORN1);
                                // }
                                // two_hop_search_ACORN_RANGE(data_point, candidate_id, vl, visitedNodes, left_range, right_range, top_candidates_ACORN1);

                                // if (result_computer_check(additionalData, candidate_id))
                                // {
                                //     top_candidates_ACORN1.emplace(dist, candidate_id);
                                //     // two_hop_search_ACORN(data_point, ep_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);
                                // }
                                // //     // int popularity_result = compute_pop_during_exhasutive(ep_id, additionalData);
                                // //     // if (popularity_result >= 10)
                                //     int popularity_result = compute_corelation_pop(candidate_id, string_vector);
                                // //   cout<<"Pop_ "<<popularity_result<<endl;
                                //     if (popularity_result >1){
                                //  two_hop_search_ACORN(data_point, candidate_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);
                                //     }
                                //     else{
                                //         one_hop_search_ACORN(data_point, candidate_id, vl, visitedNodes, additionalData, top_candidates_ACORN1);

                                //     }
                                // if(range_int_computer(left_range, right_range, candidate_id)){
                                //  top_candidates_ACORN1.emplace(dist, candidate_id);

                                // }
                                // //                       pair<double, int> CDF_Difference = predictedCDF(candidate_id,left_range, right_range, ep_id);
                                // // if(CDF_Difference.first<0.2)
                                //  two_hop_search_ACORN_RANGE(data_point, candidate_id, vl, visitedNodes, left_range, right_range, top_candidates_ACORN1);
                                // if (multidiemensional_search(candidate_id, left_range, right_range, additionalData))
                                // {
                                //     top_candidates_ACORN1.emplace(dist, candidate_id);
                                // }
                                // two_hop_search_multidiemension_ACORN(data_point, vl, additionalData, left_range, right_range, candidate_id, visitedNodes, top_candidates_ACORN1);

                                if (!bare_bone_search && stop_condition)
                                {
                                    stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                                }
                            }

                            bool flag_remove_extra = false;
                            if (!bare_bone_search && stop_condition)
                            {
                                flag_remove_extra = stop_condition->should_remove_extra();
                            }
                            else
                            {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                            while (flag_remove_extra)
                            {
                                tableint id = top_candidates.top().second;
                                top_candidates.pop();
                                if (!bare_bone_search && stop_condition)
                                {
                                    stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                    flag_remove_extra = stop_condition->should_remove_extra();
                                }
                                else
                                {
                                    flag_remove_extra = top_candidates.size() > ef;
                                }
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

        void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            const size_t M)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                     getDataByInternalId(curent_pair.second),
                                     dist_func_param_);
                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list)
            {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        linklistsizeint *get_linklist0(tableint internal_id) const
        {
            return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const
        {
            return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }

        linklistsizeint *get_linklist(tableint internal_id, int level) const
        {
            return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        }

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const
        {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        }

        tableint mutuallyConnectNewElement(
            const void *data_point,
            tableint cur_c,
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            int level,
            bool isUpdate)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                // lock only during the update
                // because during the addition the lock for cur_c is already acquired
                std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
                if (isUpdate)
                {
                    lock.lock();
                }
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {
                std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *)(ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate)
                {
                    for (size_t j = 0; j < sz_link_list_other; j++)
                    {
                        if (data[j] == cur_c)
                        {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present)
                {
                    if (sz_link_list_other < Mcurmax)
                    {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    }
                    else
                    {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                             dist_func_param_),
                                data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0)
                        {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        void resizeIndex(size_t new_max_elements)
        {
            if (new_max_elements < cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

            visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new = (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        size_t indexFileSize() const
        {
            size_t size = 0;
            size += sizeof(offsetLevel0_);
            size += sizeof(max_elements_);
            size += sizeof(cur_element_count);
            size += sizeof(size_data_per_element_);
            size += sizeof(label_offset_);
            size += sizeof(offsetData_);
            size += sizeof(maxlevel_);
            size += sizeof(enterpoint_node_);
            size += sizeof(maxM_);

            size += sizeof(maxM0_);
            size += sizeof(M_);
            size += sizeof(mult_);
            size += sizeof(ef_construction_);

            size += cur_element_count * size_data_per_element_;

            for (size_t i = 0; i < cur_element_count; i++)
            {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                size += sizeof(linkListSize);
                size += linkListSize;
            }
            return size;
        }

        void saveIndex(const std::string &location)
        {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++)
            {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0)
        {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            clear();
            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:
            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (input.tellg() < 0 || input.tellg() >= total_filesize)
                {
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0)
                {
                    input.seekg(linkListSize, input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if (input.tellg() != total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();
            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

            visited_list_pool_.reset(new VisitedListPool(1, max_elements));

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++)
            {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0)
                {
                    element_levels_[i] = 0;
                    linkLists_[i] = nullptr;
                }
                else
                {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *)malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (isMarkedDeleted(i))
                {
                    num_deleted_ += 1;
                    if (allow_replace_deleted_)
                        deleted_elements.insert(i);
                }
            }

            input.close();

            return;
        }

        template <typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataByInternalId(internalId);
            size_t dim = *((size_t *)dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *)data_ptrv;
            for (size_t i = 0; i < dim; i++)
            {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        /*
         * Marks an element with the given label deleted, does NOT really change the current graph.
         */
        void markDelete(labeltype label)
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            markDeletedInternal(internalId);
        }

        /*
         * Uses the last 16 bits of the memory for the linked list size to store the mark,
         * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
         */
        void markDeletedInternal(tableint internalId)
        {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.insert(internalId);
                }
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /*
         * Removes the deleted mark of the node, does NOT really change the current graph.
         *
         * Note: the method is not safe to use when replacement of deleted elements is enabled,
         *  because elements marked as deleted can be completely removed by addPoint
         */
        void unmarkDelete(labeltype label)
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            unmarkDeletedInternal(internalId);
        }

        /*
         * Remove the deleted mark of the node.
         */
        void unmarkDeletedInternal(tableint internalId)
        {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
                if (allow_replace_deleted_)
                {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.erase(internalId);
                }
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /*
         * Checks the first 16 bits of the memory to see if the element is marked deleted.
         */
        bool isMarkedDeleted(tableint internalId) const
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint *ptr) const
        {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint *ptr, unsigned short int size) const
        {
            *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
        }

        /*
         * Adds point. Updates the point if it is already in the index.
         * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
         */
        void addPoint(const void *data_point, labeltype label, bool replace_deleted = false)
        {
            if ((allow_replace_deleted_ == false) && (replace_deleted == true))
            {
                throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
            }

            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
            if (!replace_deleted)
            {

                addPoint(data_point, label, -1);
                return;
            }

            // check if there is vacant place
            tableint internal_id_replaced;
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
            bool is_vacant_place = !deleted_elements.empty();
            if (is_vacant_place)
            {
                internal_id_replaced = *deleted_elements.begin();
                deleted_elements.erase(internal_id_replaced);
            }
            lock_deleted_elements.unlock();

            // if there is no vacant place then add or update point
            // else add point to vacant place
            if (!is_vacant_place)
            {

                addPoint(data_point, label, -1);
            }
            else
            {
                // we assume that there are no concurrent operations on deleted element

                labeltype label_replaced = getExternalLabel(internal_id_replaced);
                setExternalLabel(internal_id_replaced, label);

                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                label_lookup_.erase(label_replaced);
                label_lookup_[label] = internal_id_replaced;
                lock_table.unlock();

                unmarkDeletedInternal(internal_id_replaced);
                updatePoint(data_point, internal_id_replaced, 1.0);
            }
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability)
        {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++)
            {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto &&elOneHop : listOneHop)
                {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto &&elTwoHop : listTwoHop)
                    {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto &&neigh : sNeigh)
                {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto &&cand : sCand)
                    {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep)
                        {
                            candidates.emplace(distance, cand);
                        }
                        else
                        {
                            if (distance < candidates.top().first)
                            {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *)(ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++)
                        {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        }

        void repairConnectionsForUpdate(
            const void *dataPoint,
            tableint entryPointInternalId,
            tableint dataPointInternalId,
            int dataPointLevel,
            int maxLevel)
        {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel)
            {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj, level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++)
                        {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist)
                            {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--)
            {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0)
                {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0)
                {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted)
                    {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level)
        {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *)(data + 1);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        }

        tableint addPoint(const void *data_point, labeltype label, int level)
        {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);

                auto search = label_lookup_.find(label);

                /// Deal Later.
                if (search != label_lookup_.end())
                {

                    tableint existingInternalId = search->second;
                    if (allow_replace_deleted_)
                    {
                        if (isMarkedDeleted(existingInternalId))
                        {
                            throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                        }
                    }
                    lock_table.unlock();

                    if (isMarkedDeleted(existingInternalId))
                    {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }
                // Debugging

                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                }
                // Update the map for each value
                // For each map[Key, Value] add cur_c which is here Label is the identifier for the vector for example it is the  increment of the loop
                cur_c = cur_element_count;
                // Update the Map here for the meta_data

                cur_element_count++;
                // Example this un ordered map has key 1 with value cur_c.
                label_lookup_[label] = cur_c;
            }

            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);

            if (level > 0)
                curlevel = level;
            // For each new data item this vector keeps it starting level. from where it start inserting
            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);

            int maxlevelcopy = maxlevel_;

            if (curlevel <= maxlevelcopy)
                templock.unlock();

            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label

            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));

            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel)
            { // if cur level is non zero then this.

                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1)
            {

                if (curlevel < maxlevelcopy)
                {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);

                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);

                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                // std::cout<<"Here"<<curlevel<<"===="<< cur_c<<"==="<<maxlevelcopy<<std::endl;
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }

            return cur_c;
        }

        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr, unsigned int start_range = 0, unsigned int end_range = 0, std::vector<char *> additionalData = std::vector<char *>()) const
        {
            std::priority_queue<std::pair<dist_t, labeltype>> result;

            if (cur_element_count == 0)
                return result;

            tableint currObj = enterpoint_node_;

            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            // From this for loop we Just move to the root level node only. Simply you can say entrypoint for level 0
            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)get_linklist(currObj, level);

                    // std::cout<<"Data"<<*data<<"Level==="<<level<<std::endl;
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    // std::cout<<"Data==="<<*datal<<std::endl;
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            bool bare_bone_search = !num_deleted_ && !isIdAllowed;

            if (bare_bone_search)
            {
                top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed, start_range, end_range, additionalData);
            }
            else
            {
                top_candidates = searchBaseLayerST<false>(

                    currObj, query_data, std::max(ef_, k), isIdAllowed, start_range, end_range, additionalData);
            }

            // while (top_candidates.size() > k)
            // {
            //     top_candidates.pop();
            // }

            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));

                top_candidates.pop();
            }

            return result;
        }

        std::vector<std::pair<dist_t, labeltype>>
        searchStopConditionClosest(
            const void *query_data,
            BaseSearchStopCondition<dist_t> &stop_condition,
            BaseFilterFunctor *isIdAllowed = nullptr) const
        {
            std::vector<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0)
                return result;

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--)
            {

                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

            size_t sz = top_candidates.size();
            result.resize(sz);
            while (!top_candidates.empty())
            {
                result[--sz] = top_candidates.top();
                top_candidates.pop();
            }

            stop_condition.filter_results(result);

            return result;
        }

        void checkIntegrity()
        {
            int connections_checked = 0;
            std::vector<int> inbound_connections_num(cur_element_count, 0);
            for (int i = 0; i < cur_element_count; i++)
            {
                for (int l = 0; l <= element_levels_[i]; l++)
                {
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *)(ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j = 0; j < size; j++)
                    {
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            if (cur_element_count > 1)
            {
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
                for (int i = 0; i < cur_element_count; i++)
                {
                    assert(inbound_connections_num[i] > 0);
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";
        }

        //

        void addPointWithMetaData(const void *data_point, labeltype label, std::vector<char *> metaData, bool replace_deleted = false)
        {
        }

        // Creating cluster, 2) in parameter pass the size of cluster for new one.
        // Count Min sketch to hold the frequency of data items
        // Bloom filter is Just populated for frequent search.
        void clustering_and_maintaining_sketch(tableint sizeOfCluster)
        {
            // Define the memory block for Disk optimization access;
            unsigned int file_count = 100000;
            unsigned int file_size = 8;
            bit_array_for_disk_access = (char *)malloc(file_count * file_size);
            memset(bit_array_for_disk_access, 0, file_count * (file_size));

            std::vector<CountMinSketchMinHash> cms;
            unsigned int cms_counter = 0;
            unsigned int clusterNumber = 1;
            cms.push_back(CountMinSketchMinHash());
            /// Correct count Map

            int counterForFilter = 0; // also for updating the cluster

            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;

            for (tableint id = 0; id < meta_data_predicates.size(); id++)
            {

                if (visitedIds.find(id) != visitedIds.end())
                    continue;

                visitedIds.insert(id);
                // Insert data
                for (const auto &predicate : meta_data_predicates[id])
                {

                    cms[cms_counter].update(predicate, id, 1);
                    // std::cout<<"New Location"<<clusterNumber-1<<std::endl;
                    // Correct map insertion
                }

                bit_manipulation_short(id, clusterNumber);

                counterForFilter++;

                int *data = (int *)get_linklist0(id);

                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);

                    for (const auto &predicate : meta_data_predicates[candidateId])
                    {

                        cms[cms_counter].update(predicate, candidateId, 1);
                    }

                    // cluster_hash_based_updating(candidateId, clusterNumber);
                    bit_manipulation(candidateId, clusterNumber);
                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                    // }

                    // // Two hop insertion
                    // for (size_t j = 0; j < size; j++)
                    // {
                    // tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        visitedIds.insert(candidateIdTwoHop);

                        for (const auto &predicate : meta_data_predicates[candidateIdTwoHop])
                        {

                            cms[cms_counter].update(predicate, candidateIdTwoHop, 1);
                        }

                        // cluster_hash_based_updating(candidateIdTwoHop, clusterNumber);
                        bit_manipulation(candidateIdTwoHop, clusterNumber);
                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }

                // Updating the data structure after insertion
                if (counterForFilter >= sizeOfCluster)
                {
                    // cout<<"counterForFilter::: "<<counterForFilter<<endl;

                    mapForCMS[clusterNumber] = cms[cms_counter]; // Store the CMS

                    // Writing for testing
                    // cms[cms_counter].saveToFile("/data3/""/CMS_size/" + std::to_string(clusterNumber) + ".bin");

                    // mapForCMS[clusterNumber].saveToFile("/data4/hnsw/paper/Clusters/CMS_256_512/" + std::to_string(clusterNumber) + ".bin");
                    clusterNumber++;
                    cms_counter++;
                    cms.push_back(CountMinSketchMinHash());
                    //  cms[clusterNumber-1]= CountMinSketchMinHash();
                    counterForFilter = 0;

                    // cms_init(&cms, count_min_width, count_min_height);         // Reinitialize the CMS
                    // bloom_filter_init(&bloom_filter, bloom_filter_size, 0.05); //  bloom_filter.clear();                                // Clear the Bloom filter (if applicable)
                }
            }
            // All  remaining insertion
            if (counterForFilter > 0)
            {

                // mapForBF[clusterNumber] = std::move(bloom_filter);
                mapForCMS[clusterNumber] = cms[cms_counter];
                //  cms[cms_counter].saveToFile("/data3/""/CMS_size/" + std::to_string(clusterNumber) + ".bin");

                // mapForCMS[clusterNumber].saveToFile("/data4/hnsw/paper/Clusters/CMS_256_512/" + std::to_string(clusterNumber) + ".bin");

                // Writing IDS
                //  writeIdsAndClusterRelationShip("/data4/hnsw/paper/Clusters//map_256_512.bin", "/data4/hnsw/paper/Clusters//short_256_512.bin");
            }
        }

        int compute_popularity(hnswlib::labeltype &id, std::vector<char *> &metaData, std::vector<std::pair<int, std::set<char> *>> &setIntersection, unsigned int &selected_cluster)
        {
            int maximum_frequency = -1;

            // #ifdef QF
            union Data data;
            data.raw.bytes[0] = mem_for_ids_clusters[id * 3];
            data.raw.bytes[1] = mem_for_ids_clusters[id * 3 + 1];
            data.raw.bytes[2] = mem_for_ids_clusters[id * 3 + 2];
            data.raw.padding = 0; // Ensure no garbage in the 4th byte

            const bool is_second_bit_set = (data.value & (1 << 1)) != 0;
            const uint32_t cluster_id = data.value >> 8;
            // cout<<"CLuster_id"<<cluster_id<<endl;

            if (is_second_bit_set)
            {
                // Process multiple clusters

                //  unsigned int cluster_number = getRandomElement(cluster_Mem_chk[id]);
                // CountMinSketchMinHash &cms_instance = mapForCMS[cluster_number];
                //  ProcessAttributes(cms_instance, metaData, maximum_frequency, setIntersection, cluster_number, selected_cluster);
                const auto &clusters = cluster_Mem_chk.at(id);

                for (const tableint cluster_number : clusters)
                {
                    CountMinSketchMinHash &cms_instance = mapForCMS.at(cluster_number);
                    ProcessAttributes(cms_instance, metaData, maximum_frequency, setIntersection, cluster_number, selected_cluster);
                    //  break;
                }
            }
            else
            {
                // Process single cluster
                CountMinSketchMinHash &cms_instance = mapForCMS.at(cluster_id);
                ProcessAttributes(cms_instance, metaData, maximum_frequency, setIntersection, cluster_id, selected_cluster);
            }
// #endif

// #ifdef array_based
//              uint32_t value_at_index = 0;
//              // Read 3 bytes into value_at_index

//             memcpy(&value_at_index, mem_for_ids_clusters + (id * 3), 3);

//             // Check if the second bit is set
//             bool is_second_bit_set = (value_at_index & (1 << 1)) != 0;

//             // Declare a variable to hold cluster ID
//             uint32_t cluster_id = value_at_index >> 8;

//   //cout<<"CLuster_id"<<cluster_id<<endl;
//             // Only proceed with the logic if the cluster ID is valid
//             if (is_second_bit_set)
//             {

//                 // Use a set to avoid duplicate calculations for attributes
//                 std::unordered_set<unsigned int> clusters = cluster_Mem_chk[id];
//                 for (tableint cluster_number : clusters)
//                 {
//                     // Use a reference to mapForCMS[cluster_number] to avoid repeated lookups
//                     auto &cms_instance = mapForCMS[cluster_number];

//                     for (auto &attribute : metaData)
//                     {
//                         if (attribute != nullptr)
//                         { // Ensure attribute is valid
//                             std::pair<unsigned int, unsigned int> pair_ = cms_instance.estimate(attribute);
//                             int res = cms_instance.C[pair_.first][pair_.second];

//                             // Update maximum frequency if needed
//                             maximum_frequency = std::max(maximum_frequency, res);
//                             // Maaking it comment for ust multidiemensional range search
//                             // setIntersection.push_back(std::make_pair(maximum_frequency, &cms_instance.min_hash_RBT[pair_.first][pair_.second]));
//                         }
//                     }
//                 }
//             }
//             else
//             {
//                 // When the second bit is not set
//                 // Use a reference to mapForCMS[cluster_id] to avoid repeated lookups
//                 auto &cms_instance = mapForCMS[cluster_id];

//                 for (auto &attribute : metaData)
//                 {
//                     if (attribute != nullptr)
//                     { // Ensure attribute is valid
//                         // Return indexes
//                         std::pair<unsigned int, unsigned int> pair_ = cms_instance.estimate(attribute);
//                         int res = cms_instance.C[pair_.first][pair_.second];

//                         // Update maximum frequency if needed
//                         maximum_frequency = std::max(maximum_frequency, res);
//                         // setIntersection.push_back(std::make_pair(maximum_frequency, &cms_instance.min_hash_RBT[pair_.first][pair_.second]));
//                     }
//                 }
//             }

// #endif
#ifdef bloom_filter

#endif
            // Find intersection of all
            // popularity_map[id]= maximum_frequency;
            return maximum_frequency;
        }

        /**
         * @brief It first find Top-K using searKNN
         * while loop iterate the priority queue.
         * call the cluster_base_searching for exhaustive one hop or two search from nearest topk
         */

        void clustered_based_exhaustive_search(const void *query_data, size_t k, std::vector<char *> &additionalData, const int &query_num, const int &efs, const int &popularity_threshold, std::string file_record)
        {

            std::unordered_set<int> visitedNodes;
            std::vector<std::pair<int, float>> result_vector;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = searchKnn(query_data, k, nullptr, 0, 0, additionalData);
            std::unordered_set<int> visited_clusters;

            // set intersection finding similar elements

            std::vector<std::pair<int, std::set<char> *>> set_intersec;

            while (!result.empty())
            {
                // Access the top element (a pair of float and labeltype)
                std::pair<float, hnswlib::labeltype> top_element = result.top();

                // Get the labeltype from the pair
                hnswlib::labeltype label = top_element.second;

                // Process the labeltype as needed

                // if identiied node matches the predicate

                if (result_computer_check(additionalData, label))
                {
                    result_vector.push_back(std::make_pair(label, top_element.first));
                }

                unsigned int selected_cluster = -1;

                // Precompute the popularity result only once
                int popularity_result = compute_popularity(label, additionalData, set_intersec, selected_cluster);

                // Check if the cluster has already been visited
                if (visited_clusters.count(selected_cluster))
                {
                    // Cluster has already been visited, perform one-hop search
                    one_hop_search(query_data, additionalData, label, visitedNodes, result_vector);
                }
                else
                {
                    // Mark the cluster as visited
                    visited_clusters.insert(selected_cluster);

                    // Find largest intersection only once
                    std::pair<int, size_t> intersection = find_largest_intersection(set_intersec);

                    // If intersection size is large, do one-hop search we 8 bit however, it can be increased
                    if (intersection.second >= 250)
                    {
                        one_hop_search(query_data, additionalData, label, visitedNodes, result_vector);
                    }
                    else
                    {
                        // Otherwise, determine the search method based on popularity result
                        if (popularity_result > popularity_threshold)
                        {
                            two_hop_search(query_data, additionalData, label, visitedNodes, result_vector);
                        }
                        else
                        {
                            one_hop_search(query_data, additionalData, label, visitedNodes, result_vector);
                        }
                    }
                }

                result.pop();
            }

            if (!result_vector.empty())
            {

                std::sort(result_vector.begin(), result_vector.end(),
                          [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                          {
                              return a.second < b.second; // Sort by the float value (distance) in ascending order
                          });

                // std::string file_path = "/data4/hnsw/paper/Recall/16_32/Q" + std::to_string(query_num) + ".csv";
                std::string file_path = file_path + std::to_string(efs) + "/Q" + std::to_string(query_num) + ".csv";

                // Open file in write mode
                std::ofstream file(file_path);

                // Check if the file is open
                if (!file.is_open())
                {
                    std::cerr << "Failed to open file: " << file_path << std::endl;
                    return;
                }

                // Write the header
                file << "ID,distance\n";

                // Write the data
                for (const auto &entry : result_vector)
                {
                    file << entry.first << "," << entry.second << "\n";
                }
                // std::cout << "First_record" << std::endl;
                // Close the file
                file.close();
            }
        }

        // disk access
        void counter_disk_access()
        {
            cout << "   counter_or_disk_access" << counter_or_disk_access << endl;
            counter_or_disk_access = 0;
        }
      

        /**
         * @brief Performs one hop or two hop search depends on the popularity
         * If popularity is higher then go for two hop.
         * Pushes the results into the result vector.
         */
        void cluster_base_searching(const void *query_data, std::vector<char *> &additionalData, hnswlib::labeltype &node,
                                    std::unordered_set<int> &visitedNodes, std::vector<std::pair<int, float>> &result_vector)
        {

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            visited_array[node] = visited_array_tag;
            // One Hop Neighbour
            // Data and its pointer
            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                if (visited_array[candidate_id] == visited_array_tag || !(visitedNodes.count(candidate_id) == 0))
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                visitedNodes.insert(candidate_id);

                if (result_computer_check(additionalData, candidate_id))
                {

                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    result_vector.push_back(std::make_pair(candidate_id, dist1));
                }
            }
            // Check condition for two hop only for those keys whose popularity is higher than certain threshold
            // Two hop Neighbour
            // It is popularity that computes the popularity of the key
            // Computing the popularity of input tuple.

            //  auto start = std::chrono::high_resolution_clock::now();

            // Execute the code you want to measure
            int popularity_result = compute_popularity(node, additionalData);

            // Stop measuring time
            // auto end = std::chrono::high_resolution_clock::now();

            // // Calculate the duration in microseconds
            // std::chrono::duration<double, std::micro> duration = end - start;

            // // Output the result
            // std::cout << "Popularity Result: " << popularity_result << std::endl;
            // std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

            // std::cout<<"Duration is"<<duration<<std::endl;
            // std::cout<<popularlty_result<<"Popularity";
            //  std::cout<<"Popularity"<<popularlty_result<<std::endl;
            // // result_counter_one_hop++;
            // // Apply condition First here.

            if (popularity_result > 300)
            {
                //     // result_counter_two_hop++;
                // std::cout << "Popularity" << popularlty_result << std::endl;
                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    if (!twoHopData)
                        continue; // Error handling

                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                        _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                        if (visited_array[candidateIdTwoHop] == visited_array_tag || !(visitedNodes.count(candidateIdTwoHop) == 0))
                            continue;
                        visited_array[candidateIdTwoHop] = visited_array_tag;
                        visitedNodes.insert(candidateIdTwoHop);
                        if (result_computer_check(additionalData, candidateIdTwoHop))
                        {
                            char *currObj1 = (getDataByInternalId(candidateIdTwoHop));

                            dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                            result_vector.push_back(std::make_pair(candidateIdTwoHop, dist1));
                        }
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }

        void cluster_hash_based_updating(const int &node_identifier, const int &cluster_identifier)
        {

            if (cluster_hash_based.find(node_identifier) != cluster_hash_based.end())
            {
                // If present, add the clusterNumber to the set
                cluster_hash_based[node_identifier].insert(cluster_identifier);
            }
            else
            {
                std::unordered_set<unsigned int> newSet;
                newSet.insert(cluster_identifier);
                cluster_hash_based[node_identifier] = newSet;
            }
        }

        /**
         * @brief Checks if any query predicate matches the nearest neighbor's predicates.
         * Iterates through the query predicates and compares them with the nearest neighbor's metadata.
         * Returns `true` if a match is found, otherwise returns `false`.
         */

        bool result_computer_check(std::vector<char *> &query_predicates, tableint nearest_neighbour_id) const
        {

            bool flag = false; // Corrected to `bool`

            for (auto &query_predicate : query_predicates)
            {
                char *pred = query_predicate;

                // bool file_check = compute_file_check(nearest_neighbour_id, pred, 1000000, 8);
                // if (file_check)
                //     counter_or_disk_access = counter_or_disk_access + 1;

                for (auto &predicate : meta_data_predicates.at(nearest_neighbour_id))
                {

                    if (strcmp(pred, predicate) == 0)
                    {
                        // std::cout << "Match found! pred: " << pred << ", predicate: " << predicate << std::endl;
                        flag = true; // Set flag to true when a match is found
                        break;       // Break out of the inner loop as the match is found
                    }
                }

                if (flag)
                {
                    break; // Break out of the outer loop once a match is found
                }
                // Break out of the outer loop once a match is found
            }
            return flag;
        }

        // Bit manupliation for short variables

        /**
         * @brief Updates a node's cluster information using bit manipulation.
         * Reads 3 bytes of data, checks if the least significant bit is set, and either updates the value or inserts a new cluster.
         * The function modifies the `mem_for_ids_clusters` and updates the `cluster_Mem_chk` map accordingly.
         */

        void bit_manipulation_short(const int &node_identifier_, const unsigned int &cluster_identifier_)
        {
            // size_t byte_offset = 3;
            uint32_t value_at_index = 0;
            memcpy(&value_at_index, mem_for_ids_clusters + (node_identifier_ * 3), 3);

            // Check the first bit (least significant bit)
            bool is_first_bit_set = (value_at_index & 1) != 0;

            if (is_first_bit_set)
            {
                //  cout<<"is irst bit iniside"<<value_at_index<<endl;
                unsigned int clusterID_ = value_at_index >> 8; // for Map Insertion when it contain more values

                // Update the hashvalue or maintain the hashvalue for clusters.
                value_at_index = value_at_index | (1 << 1);

                // value_at_index=value_at_index|(1 << clusterID);
                memcpy(mem_for_ids_clusters + (node_identifier_ * 3), &value_at_index, 3);

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
                memcpy(mem_for_ids_clusters + (node_identifier_ * 3), &value_at_index, 3);
            }
        }

        /**
         * @brief Updates a node's cluster information using bit manipulation.
         * Reads 4 bytes of data, checks if the least significant bit is set, and either updates the value or inserts a new cluster.
         * The function modifies the `mem_for_ids_clusters` and updates the `cluster_Mem_chk` map accordingly.
         */

        void bit_manipulation(const int &node_identifier_, const unsigned int &cluster_identifier_)
        {

            // size_t byte_offset = node_identifier * sizeof(unsigned int);
            // unsigned int *address_of_index = (unsigned int *)(mem_for_ids_clusters + byte_offset);

            // Dereference the pointer to get the integer value use unsigned int other wise signed may cause issue for most significant bits
            // unsigned int value_at_index = *address_of_index;
            unsigned int value_at_index = clusterIDs[node_identifier_];
            // Check the first bit (least significant bit)
            bool is_first_bit_set = (value_at_index & 1) != 0;

            if (is_first_bit_set)
            {
                unsigned int clusterID_ = value_at_index >> 16; // for Map Insertion when it contain more values

                // Update the hashvalue or maintain the hashvalue for clusters.
                value_at_index = value_at_index | (1 << 1);

                // value_at_index=value_at_index|(1 << clusterID);
                clusterIDs[node_identifier_] = value_at_index;

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
                value_at_index |= (cluster_identifier_ << 16);
                value_at_index |= (1 << 0); // set on position 0 if the array position is 0
                clusterIDs[node_identifier_] = value_at_index;
            }
        }

        // GroundTruth computer
        /* Computes the ground truth using brute force approach
        Query vectors and data sets are input
        using the distance function of the HNSWlib*/

        // std::vector<char *> &additionalData
        void ground_truth_computer_for_predicate(const void *query_data, size_t k, unsigned int &start_range, unsigned int &end_range, int &query_num) //, std::unordered_map<unsigned int, std::vector<char *>> &metaDataMap

        {
            std::vector<std::pair<int, float>> ground_truth_for_queries;

            for (int j = 0; j < max_elements_; j++)
            {
                if (range_int_computer(start_range, end_range, j))
                {
                    char *currObj1 = getDataByInternalId(j);

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                    ground_truth_for_queries.emplace_back(j, dist1);
                }

                // bool flag = false;

                // Assuming meta_data_predicates is accessible here and contains pairs of (string, vector<char*>)
                // std::pair<std::string, std::vector<char *>> pair_of_vector = meta_data_multidiemesional_query[j];

                // if (pair_of_vector.first >= start_range && pair_of_vector.first <= end_range)
                // {
                //     // cout<<"Pair"<<start_range<<"Pair se"<<end_range<<"    "<< pair_of_vector.first<<endl;

                //     ground_truth_for_queries.emplace_back(j, dist1);
                // }

                //  std::pair<std::string, std::vector<char *>> pair_of_vector = meta_data_multidiemesional_query[j];
            }

            // Sort the results based on the distance (second element in pair)
            std::sort(ground_truth_for_queries.begin(), ground_truth_for_queries.end(),
                      [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                      {
                          return a.second < b.second; // Sort by float in ascending order
                      });

            // Save results to CSV
            save_to_csv(ground_truth_for_queries, "/data4/hnsw/yt8m/GroundTruthGenre//Q" + std::to_string(query_num) + ".csv");
        }
        /*This function is saving the data to the file of ground truth*/
        void save_to_csv(const std::vector<std::pair<int, float>> &data, const std::string &filename)
        {
            std::ofstream file(filename); // Open the file

            if (file.is_open())
            {
                // Write the header (optional)
                // file << "ID,Distance\n";

                // Write the data
                int c = 0;
                file << "ID,distance\n";
                for (const auto &pair : data)
                {
                    file << pair.first << "," << pair.second << "\n"; // Each pair is written as ID,Distance
                    c++;
                    if (c > 15)
                    {
                        break;
                    }
                }

                file.close(); // Close the file
                std::cout << "Data saved to " << filename << " successfully!" << std::endl;
            }
            else
            {
                std::cerr << "Error: Unable to open file " << filename << std::endl;
            }
        }

        void one_hop_search(const void *query_data, std::vector<char *> &additionalData, hnswlib::labeltype &node,
                            std::unordered_set<int> &visitedNodes, std::vector<std::pair<int, float>> &result_vector)
        {

            // void one_hop_search(const void *query_data, std::vector<char *> &additionalData, hnswlib::labeltype &node,
            //                     std::unordered_set<int> &visitedNodes)
            // {

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            visited_array[node] = visited_array_tag;
            // One Hop Neighbour
            // Data and its pointer
            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                if (visited_array[candidate_id] == visited_array_tag || !(visitedNodes.count(candidate_id) == 0))
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                visitedNodes.insert(candidate_id);

                // if (result_computer_check(additionalData, candidate_id))
                // {
                if (finding_disk_data_access(additionalData[0], candidate_id))
                {

                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    result_vector.push_back(std::make_pair(candidate_id, dist1));
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }

        // void two_hop_search(const void *query_data, std::vector<char *> &additionalData, hnswlib::labeltype &node,
        //                     std::unordered_set<int> &visitedNodes)
        // {

        void two_hop_search(const void *query_data, std::vector<char *> &additionalData, hnswlib::labeltype &node,
                            std::unordered_set<int> &visitedNodes, std::vector<std::pair<int, float>> &result_vector)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            visited_array[node] = visited_array_tag;
            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                if (visited_array[candidate_id] == visited_array_tag || !(visitedNodes.count(candidate_id) == 0))
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                visitedNodes.insert(candidate_id);

                // if (finding_disk_data_access(additionalData[0], candidate_id))
                // {
                if (result_computer_check(additionalData, candidate_id))
                {

                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    result_vector.push_back(std::make_pair(candidate_id, dist1));
                }
                // Two hop searching
                int *twoHopData = (int *)get_linklist0(candidate_id);
                if (!twoHopData)
                    continue; // Error handling

                size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    if (visited_array[candidateIdTwoHop] == visited_array_tag || !(visitedNodes.count(candidateIdTwoHop) == 0))
                        continue;
                    visited_array[candidateIdTwoHop] = visited_array_tag;
                    visitedNodes.insert(candidateIdTwoHop);
                    if (result_computer_check(additionalData, candidateIdTwoHop))
                    {
                        // if (finding_disk_data_access(additionalData[0], candidateIdTwoHop))
                        // {
                        char *currObj1 = (getDataByInternalId(candidateIdTwoHop));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                        result_vector.push_back(std::make_pair(candidateIdTwoHop, dist1));
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }
        // Finding the intersection between sets and returning the greatest here.
        std::pair<int, size_t> find_largest_intersection(const std::vector<std::pair<int, std::set<char> *>> &set_intersection)
        {
            if (set_intersection.size() < 256)
            {
                return {0, 0}; // Return a default pair if there are fewer than 2 sets
            }

            // Get the last two sets in the vector
            const std::set<char> *last_set = set_intersection.back().second;
            const std::set<char> *second_last_set = set_intersection[set_intersection.size() - 2].second;

            std::set<char> intersection;

            // Perform set intersection
            std::set_intersection(
                second_last_set->begin(), second_last_set->end(),
                last_set->begin(), last_set->end(),
                std::inserter(intersection, intersection.begin()));

            // Return the pair with the int value of the second-last set and the size of the intersection

            return {set_intersection[set_intersection.size() - 2].first, intersection.size()};
        }

        void freeMemory()
        {
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

            // Iterate through mapForBF
            for (auto &pair : mapForBF)
            {
                BloomFilter &bf = pair.second;
                bloom_filter_destroy(&bf); // Destroy the BloomFilter object
            }
            mapForBF.clear(); // Clear the map after destruction

            for (auto it = cluster_and_associated_tree.begin(); it != cluster_and_associated_tree.end(); ++it)
            {
                delete it->second; // Delete the BLinkTree pointer
            }
            cluster_and_associated_tree.clear();

            // Cleaning Metadata by deallocating its pointer it is specially for Multidiemensional query
            for (auto &[key, value_pair] : meta_data_multidiemesional_query)
            {
                // Deallocate each char* in the first vector

                // Deallocate each char* in the second vector
                for (char *ptr : value_pair.second)
                {
                    delete[] ptr;
                }

                // Clear the vectors to release internal resources

                value_pair.second.clear();
            }
            // Clear the map itself

            meta_data_multidiemesional_query.clear();
            // Clear the memory allocated during disk optimization:
            free(bit_array_for_disk_access);
        }
        // Write the map to the
        void writeIdsAndClusterRelationShip(const std::string &clusterFile, const std::string &shortFile)
        {
            std::ofstream c_file(clusterFile, std::ios::binary);

            if (!c_file.is_open())
            {
                std::cerr << "Failed to open the file for writing." << std::endl;
                return;
            }

            // Write the number of entries in the unordered_map
            unsigned int map_size = cluster_Mem_chk.size();
            c_file.write(reinterpret_cast<const char *>(&map_size), sizeof(map_size));

            // Iterate through each map entry and write the data
            for (const auto &pair : cluster_Mem_chk)
            {
                // Write the key (cluster ID)
                unsigned int key = pair.first;
                c_file.write(reinterpret_cast<const char *>(&key), sizeof(key));

                // Write the size of the unordered_set
                unsigned int set_size = pair.second.size();
                c_file.write(reinterpret_cast<const char *>(&set_size), sizeof(set_size));

                // Write each element of the unordered_set
                for (const auto &value : pair.second)
                {
                    c_file.write(reinterpret_cast<const char *>(&value), sizeof(value));
                }
            }

            c_file.close();

            std::ofstream s_file(shortFile, std::ios::binary);
            if (!s_file)
            {
                std::cerr << "Failed to open file for writing!" << std::endl;
                //  delete[] mem_for_ids_clusters;
                return;
            }

            // Write the size first
            // outFile.write(reinterpret_cast<const char *>(&dataSize), sizeof(dataSize));
            s_file.write(mem_for_ids_clusters, max_elements_ * (3 * sizeof(char)));

            s_file.close();
            free(mem_for_ids_clusters);
        }

        // Read Ids of map
        void readIdsAndClusterRelationShip(const std::string &cluster_file, const std::string &short_file)
        {
            std::ifstream c_file(cluster_file, std::ios::binary); // Open in binary mode

            if (!c_file.is_open())
            {
                std::cerr << "Failed to open the file for reading." << std::endl;
                return;
            }

            // Read the number of entries in the unordered_map
            unsigned int map_size;
            c_file.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));

            // Loop through the map entries
            for (unsigned int i = 0; i < map_size; ++i)
            {
                unsigned int key;
                c_file.read(reinterpret_cast<char *>(&key), sizeof(key)); // Read key (cluster ID)

                // Read the size of the unordered_set
                unsigned int set_size;
                c_file.read(reinterpret_cast<char *>(&set_size), sizeof(set_size));

                // Read each value in the unordered_set
                std::unordered_set<unsigned int> values;
                for (unsigned int j = 0; j < set_size; ++j)
                {
                    unsigned int value;
                    c_file.read(reinterpret_cast<char *>(&value), sizeof(value)); // Read values
                    values.insert(value);
                }

                // Insert the key and its associated set into the map
                cluster_Mem_chk[key] = values;
            }

            c_file.close(); // Close the file
            // Read the short values
            std::ifstream s_File(short_file, std::ios::binary);
            // Open the file in binary mode

            if (!s_File)
            {
                std::cerr << "Error opening file for reading!" << std::endl;
                return;
            }

            // Read the memory block from the file
            s_File.read(mem_for_ids_clusters, max_elements_ * (3 * sizeof(char)));

            // Close the file
            s_File.close();
        }
        // Perform the range search
        void rangeSearch(const void *query_data, size_t k, unsigned int &start_range, unsigned int &end_range, const int &query_num, const int &efs)
        {
            std::unordered_set<int> visited_clusters;
            std::unordered_set<int> visitedNodes;
            std::vector<std::pair<int, float>> result_vector;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = searchKnn(query_data, k, nullptr, start_range, end_range);
            // Visited IDs
            double start_range_double = static_cast<double>(start_range);
            double end_range_double = static_cast<double>(end_range);
            while (!result.empty())
            {
                // Access the top element (a pair of float and labeltype)
                std::pair<float, hnswlib::labeltype> top_element = result.top();

                // Get the labeltype from the pair
                hnswlib::labeltype label = top_element.second;

                //  int count_Total = 0;
                if (range_int_computer(start_range, end_range, label))
                {
                    // if (range_result_computer_check(start_range, end_range, label))
                    // {
                    //     // The item
                    result_vector.push_back(std::make_pair(label, top_element.first));
                }
                //  }
                // two_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                // else
                // {
                pair<double, int> CDF_Difference = predictedCDF(label, start_range_double, end_range_double, query_num);

                // if(CDF_Difference.first>0.3)
                //       cout<<CDF_Difference.first<<"Time"<<endl;

                // if (visited_clusters.find(CDF_Difference.second) != visited_clusters.end())
                // {
                //     // continue
                //     // Cluster has already been visited
                //     one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                //     result.pop();
                //     continue;
                // }
                // else
                // {
                //     visited_clusters.insert(CDF_Difference.second);
                if (CDF_Difference.first < 2.5)
                {
                    two_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                }
                else
                {

                    one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                }
                // }
                //  }
                // cout<<" CDF"<< CDF_Difference.first<< endl;
                result.pop();
            }

            //         Vertex<std::string> *vertex_greater = searchMap[label]->getPrev();
            //         // while (vertex_greater)
            //         // {

            //         //     if (end_range >= vertex_greater->getCurrentNodePredicate())
            //         //     {
            //         //         // std::cout<<"Current"<<vertex_greater->getCurrentNodePredicate()<<"End   "<<end_range<<std::endl;
            //         //         char *currObj1 = (getDataByInternalId(vertex_greater->getNodeId()));
            //         //         dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
            //         //         result_vector.push_back(std::make_pair(vertex_greater->getNodeId(), dist1));
            //         //         vertex_greater = vertex_greater->getPrev();
            //         //     }
            //         //     else
            //         //     {
            //         //         break;
            //         //     }
            //         // }
            //         auto process_greater_lambda = [&](Vertex<std::string> *vertex_greater)
            //         {
            //             while (vertex_greater)
            //             {
            //                 if (end_range >= vertex_greater->getCurrentNodePredicate())
            //                 {
            //                     char *currObj1 = this->getDataByInternalId(vertex_greater->getNodeId());
            //                     dist_t dist1 = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);

            //                     {
            //                         std::lock_guard<std::mutex> lock(result_mutex);
            //                         result_vector.push_back(std::make_pair(vertex_greater->getNodeId(), dist1));
            //                     }

            //                     vertex_greater = vertex_greater->getPrev();
            //                 }
            //                 else
            //                 {
            //                     break;
            //                 }
            //             }
            //         };

            //         auto process_smaller_lambda = [&](Vertex<std::string> *vertex_smaller)
            //         {
            //             while (vertex_smaller)
            //             {
            //                 if (start_range <= vertex_smaller->getCurrentNodePredicate())
            //                 {
            //                     char *currObj1 = this->getDataByInternalId(vertex_smaller->getNodeId());
            //                     dist_t dist1 = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);

            //                     {
            //                         std::lock_guard<std::mutex> lock(result_mutex);
            //                         result_vector.push_back(std::make_pair(vertex_smaller->getNodeId(), dist1));
            //                     }

            //                     vertex_smaller = vertex_smaller->getNext();
            //                 }
            //                 else
            //                 {
            //                     break;
            //                 }
            //             }
            //         };

            //         Vertex<std::string> *vertex_smaller = searchMap[label]->getNext();

            //         std::thread t1(process_greater_lambda, vertex_greater);
            //         std::thread t2(process_smaller_lambda, vertex_smaller);

            //         // Wait for both threads to complete
            //         t1.join();
            //         t2.join();

            //         // while (vertex_smaller)
            //         // {
            //         //     if (start_range <= vertex_smaller->getCurrentNodePredicate())
            //         //     {
            //         //         std::cout << "Current" << vertex_smaller->getCurrentNodePredicate() << "End   " << start_range << std::endl;
            //         //         char *currObj1 = (getDataByInternalId(vertex_smaller->getNodeId()));
            //         //         dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
            //         //         result_vector.push_back(std::make_pair(vertex_smaller->getNodeId(), dist1));
            //         //         vertex_smaller = vertex_smaller->getNext();
            //         //     }
            //         //     else
            //         //     {
            //         //         break;
            //         //     }
            //         // }

            //         // std::string pred_str(meta_data_predicates[label][0]);
            //         // // std::cout<<"PredicateAfter"<<pred_str<<std::endl;
            //         // RangeSearch<std::string> range = searchMap[pred_str];
            //         // for (size_t ids : range.getCurrNodes())
            //         // {
            //         //     // std::cout<<"Prev-ID"<< ids<<"Range_PreV"<<range_prev.getRangeData()<<"Start"<<start_range<<std::endl;

            //         //     char *currObj1 = (getDataByInternalId(ids));

            //         //     dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

            //         //     result_vector.push_back(std::make_pair(ids, dist1));
            //         // }

            //         // RangeSearch<std::string> range_prev = searchMap[range.get_prev_node_data()];
            //         // RangeSearch<std::string> range_next = searchMap[range.get_next_node_data()];
            //         // std::string rangePrev = range_prev.getRangeData();
            //         // std::string rangeNext = range_next.getRangeData();
            //         // while (range_prev.getRangeData() >= start_range)
            //         // {

            //         //     for (size_t ids : range_prev.getCurrNodes())
            //         //     {
            //         //         // std::cout<<"Prev-ID"<< ids<<"Range_PreV"<<range_prev.getRangeData()<<"Start"<<start_range<<std::endl;

            //         //         char *currObj1 = (getDataByInternalId(ids));

            //         //         dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

            //         //         result_vector.push_back(std::make_pair(ids, dist1));
            //         //     }

            //         //     // if ( (range_prev.get_prev_node_data() == ""))
            //         //     //     break;
            //         //     range_prev = searchMap[range_prev.get_prev_node_data()];
            //         // }

            //         // while (end_range >= range_next.getRangeData())
            //         // {

            //         //     for (size_t ids : range_next.getCurrNodes())
            //         //     {
            //         //         char *currObj1 = (getDataByInternalId(ids));

            //         //         dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

            //         //         result_vector.push_back(std::make_pair(ids, dist1));
            //         //     }

            //         //     // if ((range_next.get_next_node_data() == ""))
            //         //     //     break;
            //         //     range_next = searchMap[range_next.get_next_node_data()];
            //         // }

            //         break;
            //     }

            //     result.pop();
            // }
            // //  std::cout << "Iter" << query_num << std::endl;

            // //  exit(0);
            // //  std::cout<<"Results"<<result_vector.size()<<"Query Nume"<<query_num<<std::endl;

            if (!result_vector.empty())
            {
                std::sort(result_vector.begin(), result_vector.end(),
                          [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                          {
                              return a.second < b.second; // Sort by the float value (distance) in ascending order
                          });

                std::string file_path = "/data4/hnsw/yt8m/Res_Gener/" + std::to_string(efs) + "/Q" + std::to_string(query_num) + ".csv";

                // Open file in write mode
                std::ofstream file(file_path);

                // Check if the file is open
                if (!file.is_open())
                {
                    std::cerr << "Failed to open file: " << file_path << std::endl;
                    return;
                }

                // Write the header
                file << "ID,distance\n";

                // Write the data
                int tmp_count = 0;
                for (const auto &entry : result_vector)
                {
                    file << entry.first << "," << entry.second << "\n";
                    // tmp_count= tmp_count+1;
                    // if(tmp_count==15) break;
                }
                // std::cout << "First_record" << std::endl;
                // Close the file
                file.close();
            }
        }
        // Check the ranges if the item is in between lower and upper bound
        bool range_result_computer_check(std::string &start_range, std::string &end_range, tableint nearest_neighbour_id) const
        {
            //  std::cout<<"ITems"<<start_range<<"EndRange"<<end_range<<std::endl;

            const auto &pair_of_vector = meta_data_multidiemesional_query.at(nearest_neighbour_id);

            // Directly compare the first element of the pair to the range
            return (pair_of_vector.first >= start_range && pair_of_vector.first <= end_range);
        }
        // Helper method for dealing with ranges.

        void clustering_for_cdf_range_filtering(tableint sizeOfCluster)
        {

            unsigned int clusterNumber = 1;
            // Here vector tree that insert the vector data for CDDF
            std::vector<std::pair<tableint, std::string>> predicate_data_CDF;
            int counterForFilter = 0; // also for updating the cluster

            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;
            // meta_data_multi_diemensional_qery here used as meta data.

            for (tableint id = 0; id < meta_data_multidiemesional_query.size(); id++)
            {

                if (visitedIds.find(id) != visitedIds.end())
                    continue;
                visitedIds.insert(id);
                // Insert data for CDF computation here.
                ////////  predicate_data_CDF.emplace_back(id, meta_data_multidiemesional_query[id].first);
                // Do addition for Count_min_sketch.

                bit_manipulation_short(id, clusterNumber);

                counterForFilter++;

                int *data = (int *)get_linklist0(id);
                if (!data)
                    continue; // Error handling

                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);

                    // Inserting the CDF Value here
                    predicate_data_CDF.emplace_back(candidateId, meta_data_multidiemesional_query[candidateId].first);

                    // To Do insert Count min sketch Logic
                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                }

                // Two hop insertion
                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    if (!twoHopData)
                        continue; // Error handling

                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        visitedIds.insert(candidateIdTwoHop);
                        // Insert the tree logic
                        predicate_data_CDF.emplace_back(candidateIdTwoHop, meta_data_multidiemesional_query[candidateIdTwoHop].first);
                        // Todo the CMS value

                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }

                // Updating the data structure after insertion
                if (counterForFilter >= sizeOfCluster)
                {

                    std::sort(predicate_data_CDF.begin(), predicate_data_CDF.end(),
                              [](const std::pair<tableint, std::string> &a, const std::pair<tableint, std::string> &b)
                              {
                                  return a.second < b.second;
                              });

                    // Computing the CDF
                    // Step 1: Calculate the total number of data points
                    // Step 1: Calculate the total number of data points

                    size_t n = predicate_data_CDF.size();

                    // Counting the occurance of unique values and populate the vector

                    std::map<std::string, std::vector<tableint>> count_map;
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
                    std::vector<std::pair<std::string, double>> cdf; // (value, cumulative probability)

                    // Step 4: Compute the CDF
                    double cumulative_count = 0;
                    // Create a Map here that where we insert the ID and the set that contain the range identifier and
                    // set that contain it
                    std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;
                    // Here Count is the complete vector of IDs
                    for (const auto &[value, count] : count_map)
                    {
                        cumulative_count += count.size(); // Increment cumulative count

                        double cumulative_probability = cumulative_count / n; // Calculate cumulative probability
                        // Checking the probablity and range for K MinWise hashing
                        int label = check_range_search(cumulative_probability);
                        //  int labelTesting=check_range_search(cumulative_probability);
                        //  cout<<"Label"<<label<<"==="<<labelTesting<<"  "<< cumulative_probability<<endl;
                        computing_and_inserting_relevant_range(label, count, map_cdf_range_k_minwise);

                        cdf.push_back({value, cumulative_probability});
                    }

                    /// Compute the Regression Model using Eign
                    // Insert it into the  map insert it into the map[id, Model]
                    //  Flush the predicate Vector

                    RegressionModel reg_model;

                    reg_model.train(cdf);
                    reg_model.setMapCdfRangeKMinwise(map_cdf_range_k_minwise);
                    reg_model.save("/data3/"
                                   "/CMS_size/" +
                                   std::to_string(clusterNumber) + ".bin");

                    mapForRegressionModel[clusterNumber] = std::move(reg_model);

                    clusterNumber++;
                    counterForFilter = 0;
                    predicate_data_CDF = std::vector<std::pair<tableint, std::string>>();
                    cdf = std::vector<std::pair<std::string, double>>();

                    /// Map for entering into the

                    // cms_init(&cms, count_min_width, count_min_height);         // Reinitialize the CMS
                    // bloom_filter_init(&bloom_filter, bloom_filter_size, 0.05); //  bloom_filter.clear();                                // Clear the Bloom filter (if applicable)
                }
            }
            // All  remaining insertion
            if (counterForFilter > 0)
            {
                std::sort(predicate_data_CDF.begin(), predicate_data_CDF.end(),
                          [](const std::pair<tableint, std::string> &a, const std::pair<tableint, std::string> &b)
                          {
                              return a.second < b.second;
                          });
                // Computing the CDF
                // Step 1: Calculate the total number of data points
                // Step 1: Calculate the total number of data points
                size_t n = predicate_data_CDF.size();

                // Step 2: Count occurrences of each unique value// this computation is only for CDF computation
                std::map<std::string, std::vector<tableint>> count_map;
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
                std::vector<std::pair<std::string, double>> cdf; // (value, cumulative probability)

                // Step 4: Compute the CDF
                double cumulative_count = 0;
                //  here count is the size of ids vector
                std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;
                for (const auto &[value, count] : count_map)
                {
                    cumulative_count += count.size(); // Increment cumulative count

                    double cumulative_probability = cumulative_count / n; // Calculate cumulative probability
                    int label = check_range_search(cumulative_probability);

                    computing_and_inserting_relevant_range(label, count, map_cdf_range_k_minwise);

                    cdf.push_back({value, cumulative_probability});
                }
                /// Compute the Regression Model using Eign
                // Insert it into the  map insert it into the map[id, Model]
                //  Flush the predicate Vector
                // Also in the map also insert the CMS here.

                RegressionModel reg_model;
                reg_model.train(cdf);
                reg_model.setMapCdfRangeKMinwise(map_cdf_range_k_minwise);
                reg_model.save("/data3/"
                               "/CMS_size/" +
                               std::to_string(clusterNumber) + ".bin");

                mapForRegressionModel[clusterNumber] = std::move(reg_model);
                clusterNumber++;
                counterForFilter = 0;
                predicate_data_CDF = std::vector<std::pair<tableint, std::string>>();
                cdf = std::vector<std::pair<std::string, double>>();
            }
        }

        void postFilteringApproachesRange(const void *query_data, size_t k, std::string &start_range, std::string &end_range, const int &query_num, const int &efs)
        {

            std::unordered_set<int> visitedNodes;
            std::vector<std::pair<int, float>> result_vector;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = searchKnn(query_data, k);
            // exit(0);
            // int one_hop_counter_range = 0;
            // int two_hop_counter_range = 0;

            std::unordered_set<int> visited_Clusters;

            // Set intersection
            //  std::unordered_map<int, std::set<char>> set_intersec;
            // cout<<"CDFFF"<<  result.size()<<endl;
            double left_range = static_cast<double>(convertDateStringToTimestamp(start_range));
            double right_range = static_cast<double>(convertDateStringToTimestamp(end_range));

            while (!result.empty())
            {
                // Access the top element (a pair of float and labeltype)
                std::pair<float, hnswlib::labeltype> top_element = result.top();

                // Get the labeltype from the pair
                hnswlib::labeltype label = top_element.second;
                // std::cout<<label<<"Label"<<std::endl;

                if (range_result_computer_check(start_range, end_range, label))
                {
                    // The item
                    result_vector.push_back(std::make_pair(label, top_element.first));

                    // std::string pred_str(meta_data_predicates[label][0]);
                }
                // CDF Approach

                pair<double, int> CDF_Difference = predictedCDF(label, left_range, right_range, query_num);
                // cout << CDF_Difference.first << endl;
                // double CDF_Difference=predicted_CDF_map[query_num].first;
                // // //////// one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                // cout<<"CDF:::"<<CDF_Difference.first<<endl;
                if (CDF_Difference.first < 0.7)
                {
                    // if (visited_Clusters.find(CDF_Difference.second) == visited_Clusters.end())
                    // {
                    //     // //     //     // Key does not exist
                    //     visited_Clusters.insert(CDF_Difference.second);
                    two_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                }
                else
                {
                    one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                }
                // }
                // else
                // {
                //     //         // Key exists
                //     one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                // }

                // two_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                // two_hop_counter_range = two_hop_counter_range + 1;
                // }

                // else
                // {
                //     one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                //     //    // one_hop_counter_range = one_hop_counter_range + 1;
                // }

                // if (query_num <600)
                // {
                //     two_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                //     two_hop_counter_range = two_hop_counter_range + 1;
                // }

                // else
                // {
                //     one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                //     one_hop_counter_range = one_hop_counter_range + 1;
                // }

                result.pop();
            }

            // Results Writing

            if (!result_vector.empty())
            {
                std::sort(result_vector.begin(), result_vector.end(),
                          [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                          {
                              return a.second < b.second; // Sort by the float value (distance) in ascending order
                          });

                //   one_hop_two_hop.emplace_back(one_hop_counter_range, two_hop_counter_range);
                //   for(auto & range:one_hop_two_hop){
                //    std::cout<<query_num <<"   One_hop==="<<one_hop_counter_range<<"Two_hop=== "<<two_hop_counter_range<<std::endl;
                //     break;
                //   }

                std::string file_path = "/data4/hnsw/yt8m/Res_Gener/" + std::to_string(efs) + "/Q" + std::to_string(query_num) + ".csv";

                // Open file in write mode
                std::ofstream file(file_path);

                // Check if the file is open
                if (!file.is_open())
                {
                    std::cerr << "Failed to open file: " << file_path << std::endl;
                    return;
                }

                // Write the header
                file << "ID,distance\n";

                // file << one_hop_counter_range << "," << two_hop_counter_range << "\n";

                // Write the data
                for (const auto &entry : result_vector)
                {
                    file << entry.first << "," << entry.second << "\n";
                }
                //  std::cout << "First_record" << std::endl;
                // Close the file
                file.close();
            }
        }
        // pair<double, int> predictedCDF(hnswlib::labeltype &id, double &left_range, double &right_range, const int &query_id)
        pair<double, int> predictedCDF(const hnswlib::labeltype &id, double &left_range, double &right_range, const int &query_id)
        {

            double maximumFrequency = 0.0;
            int cluster_selection = -1;
            uint32_t value_at_index = 0;
            // Read 3 bytes into value_at_index

            memcpy(&value_at_index, mem_for_ids_clusters + (id * 3), 3);

            // Check if the second bit is set
            bool is_second_bit_set = (value_at_index & (1 << 1)) != 0;

            // Declare a variable to hold cluster ID
            uint32_t cluster_id = value_at_index >> 8;

            // Only proceed with the logic if the cluster ID is valid
            if (is_second_bit_set)
            {

                // Use a set to avoid duplicate calculations for attributes
                std::unordered_set<unsigned int> clusters = cluster_Mem_chk.at(id);

                for (tableint cluster_number : clusters)
                {
                    // Use a reference to mapForCMS[cluster_number] to avoid repeated lookups

                    auto &regression_Model = mapForRegressionModel.at(cluster_number);
                    // cout<<regression_Model.predict(left_range)<<"Left "<<regression_Model.predict(right_range)<<endl;
                    double CDFs = regression_Model.predict(left_range) - regression_Model.predict(right_range);
                    // Update maximum frequency if needed
                    if (CDFs < 0)
                    {
                        CDFs = -CDFs; // Convert negative to positive by multiplying by -1
                    }
                    // std::cout<<"LeftRange"<<regression_Model.predict(left_range)<<"RightRange"<<regression_Model.predict(right_range)<<std::endl;
                    if (CDFs > maximumFrequency)
                    {
                        maximumFrequency = CDFs;
                        cluster_selection = cluster_number;
                    }
                    maximumFrequency = std::max(maximumFrequency, CDFs);

                    // break;
                }
                // return std::make_pair(maximumFrequency, cluster_selection);
            }
            else
            {
                // When the second bit is not set
                // Use a reference to mapForCMS[cluster_id] to avoid repeated lookups
                auto &regression_Model = mapForRegressionModel.at(cluster_id);

                double CDFs = regression_Model.predict(left_range) - regression_Model.predict(right_range);
                // Update maximum frequency if needed
                if (CDFs < 0)
                {
                    CDFs = -CDFs; // Convert negative to positive by multiplying by -1
                }

                maximumFrequency = std::max(maximumFrequency, CDFs);
                cluster_selection = cluster_id;
                // return std::make_pair(std::max(maximumFrequency, CDFs), cluster_selection);
            }
            // predicted_CDF_map[query_id] = std::make_pair(maximumFrequency, cluster_selection);

            return std::make_pair(maximumFrequency, cluster_selection);
        }

        void one_hop_search_range(const void *query_data, unsigned int &start_range, unsigned int &end_range, hnswlib::labeltype &node,
                                  std::unordered_set<int> &visitedNodes, std::vector<std::pair<int, float>> &result_vector)
        {

            // void one_hop_search(const void *query_data, std::vector<char *> &additionalData, hnswlib::labeltype &node,
            //                     std::unordered_set<int> &visitedNodes)
            // {

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            visited_array[node] = visited_array_tag;
            // One Hop Neighbour
            // Data and its pointer
            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                if (visited_array[candidate_id] == visited_array_tag || !(visitedNodes.count(candidate_id) == 0))
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                visitedNodes.insert(candidate_id);
                if (range_int_computer(start_range, end_range, candidate_id))
                {
                    // if (range_result_computer_check(start_range, end_range, candidate_id))
                    // {

                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    result_vector.push_back(std::make_pair(candidate_id, dist1));
                }

                // if (range_result_computer_check(start_range, end_range, candidate_id))
                // {

                //     char *currObj1 = (getDataByInternalId(candidate_id));

                //     dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                //     result_vector.push_back(std::make_pair(candidate_id, dist1));
                // }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }

        void two_hop_search_range(const void *query_data, unsigned int &start_range, unsigned int &end_range, hnswlib::labeltype &node,
                                  std::unordered_set<int> &visitedNodes, std::vector<std::pair<int, float>> &result_vector)
        {

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            visited_array[node] = visited_array_tag;
            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                if (visited_array[candidate_id] == visited_array_tag || !(visitedNodes.count(candidate_id) == 0))
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                visitedNodes.insert(candidate_id);
                if (range_int_computer(start_range, end_range, candidate_id))
                {
                    // if (range_result_computer_check(start_range, end_range, candidate_id))
                    // {

                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    result_vector.emplace_back(candidate_id, dist1);
                }
                // Two hop searching
                int *twoHopData = (int *)get_linklist0(candidate_id);
                if (!twoHopData)
                    continue; // Error handling

                size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    if (visited_array[candidateIdTwoHop] == visited_array_tag || !(visitedNodes.count(candidateIdTwoHop) == 0))
                        continue;
                    visited_array[candidateIdTwoHop] = visited_array_tag;
                    visitedNodes.insert(candidateIdTwoHop);
                    // if (range_result_computer_check(start_range, end_range, candidateIdTwoHop))
                    // {
                    if (range_int_computer(start_range, end_range, candidateIdTwoHop))
                    {
                        char *currObj1 = (getDataByInternalId(candidateIdTwoHop));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                        result_vector.emplace_back(candidateIdTwoHop, dist1);
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }

        void test()
        {

            std::string date = "2020-02-23";
            for (auto &[clusterNumber, regressionModel] : mapForRegressionModel)
            {
                // Predict using the model for the given date
                double prediction = regressionModel.predict(date);

                // Print the results (cluster number and corresponding prediction)
                std::cout << "Cluster: " << clusterNumber << ", Prediction for " << date << ": " << prediction << std::endl;
                int label = check_range_search(prediction);
                std::cout << "Label" << label << endl;
            }
        }

        // Process Greater
        void process_greater(Vertex<std::string> *vertex_greater,
                             std::string end_range,
                             char *query_data,
                             void *dist_func_param_,
                             std::vector<std::pair<hnswlib::labeltype, dist_t>> &result_vector,
                             DISTFUNC<dist_t> fstdistfunc_)
        {

            while (vertex_greater)
            {
                if (end_range >= vertex_greater->getCurrentNodePredicate())
                {
                    char *currObj1 = (getDataByInternalId(vertex_greater->getNodeId()));
                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    // Lock when updating shared resource
                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        result_vector.push_back(std::make_pair(vertex_greater->getNodeId(), dist1));
                    }

                    vertex_greater = vertex_greater->getPrev();
                }
                else
                {
                    break;
                }
            }
        }
        // Process Smaller
        void process_smaller(Vertex<std::string> *vertex_smaller,
                             std::string start_range,
                             char *query_data,
                             void *dist_func_param_,
                             std::vector<std::pair<hnswlib::labeltype, dist_t>> &result_vector,
                             DISTFUNC<dist_t> fstdistfunc_)
        {

            while (vertex_smaller)
            {
                if (start_range <= vertex_smaller->getCurrentNodePredicate())
                {
                    char *currObj1 = (getDataByInternalId(vertex_smaller->getNodeId()));
                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    // Lock when updating shared resource
                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        result_vector.push_back(std::make_pair(vertex_smaller->getNodeId(), dist1));
                    }

                    vertex_smaller = vertex_smaller->getNext();
                }
                else
                {
                    break;
                }
            }
        }

        // Helper method that is generating the cluster where each cluster has a BPlus Tree that further can be used
        // For accessing the elements

        void clustering_for_bPlustree_range_filtering(tableint sizeOfCluster, SpaceInterface<dist_t> *s)
        {

            unsigned int clusterNumber = 1;
            // Here vector tree that insert the vector data
            std::vector<std::string> predicate_data;
            int counterForFilter = 0; // also for updating the cluster
            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;

            BLinkTree<std::string, unsigned int> *tree = new BLinkTree<std::string, unsigned int>(8, s);

            for (tableint id = 0; id < meta_data_predicates.size(); id++)
            {
                if (visitedIds.find(id) != visitedIds.end())
                    continue;
                visitedIds.insert(id);
                // Insert data
                for (const auto &predicate : meta_data_predicates[id])
                {

                    std::string pred_str(predicate);
                    tree->insert(pred_str, id);
                }

                bit_manipulation_short(id, clusterNumber);

                counterForFilter++;

                int *data = (int *)get_linklist0(id);
                if (!data)
                    continue; // Error handling

                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);

                    for (const auto &predicate : meta_data_predicates[candidateId])
                    {
                        std::string pred_str(predicate);
                        tree->insert(pred_str, candidateId);
                    }

                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                }

                // Two hop insertion
                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    if (!twoHopData)
                        continue; // Error handling

                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        visitedIds.insert(candidateIdTwoHop);

                        for (const auto &predicate : meta_data_predicates[candidateIdTwoHop])
                        {
                            std::string pred_str(predicate);
                            tree->insert(pred_str, candidateIdTwoHop);
                        }
                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }

                // Updating the data structure after insertion
                if (counterForFilter >= sizeOfCluster)
                {

                    cluster_and_associated_tree[clusterNumber] = tree;

                    tree = new BLinkTree<std::string, unsigned int>(8, s);

                    clusterNumber++;

                    counterForFilter = 0;
                }
            }
            // All  remaining insertion
            if (counterForFilter > 0)
            {
                cluster_and_associated_tree[clusterNumber] = tree;
                clusterNumber++;
                counterForFilter = 0;
            }
        }

        void postFiltering_range_bPlus_tree(const void *query_data, size_t k, std::string &start_range, std::string &end_range, int &query_num)
        {

            std::unordered_set<int> visitedNodes;
            std::vector<std::pair<int, float>> result_vector;
            std::vector<std::pair<dist_t, labeltype>> result = this->searchKnnCloserFirst(query_data, k);

            std::unordered_set<int> visited_ids;
            std::unordered_set<int> visited_clusters;

            std::pair<float, hnswlib::labeltype> top_element = result[0];
            hnswlib::labeltype label = top_element.second;

            predicted_bPlus_tree(label, start_range, end_range, visited_ids, result_vector, visited_clusters, query_data, query_num);
            // break;
            //  if (range_result_computer_check(start_range, end_range, label))
            //  {
            //      result_vector.push_back(std::make_pair(label, top_element.first));

            //     visted_ids.insert(label);
            //     //  predicted_bPlus_tree(label, start_range, end_range, visted_ids, result_vector, query_data);
            //     //   result_vector.push_back(std::make_pair(label, top_element.first));

            //     // std::string pred_str(meta_data_predicates[label][0]);
            //     ///  std::cout<<"Index_Number"<<query_num<<std::endl;
            //     // exit(-1);
            // }

            // //CDF Approach

            /// one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);

            //  result.pop();
            // }

            // Results Writing

            if (!result_vector.empty())
            {
                // std::cout << "Index_Number" << query_num << std::endl;
                std::sort(result_vector.begin(), result_vector.end(),
                          [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                          {
                              return a.second < b.second; // Sort by the float value (distance) in ascending order
                          });
                // std::cout << "Index_Number" << result_vector.size()<< std::endl;

                std::string file_path = "/home/u6059148/Result/PostFiltering/Q" + std::to_string(query_num) + ".csv";

                // Open file in write mode
                std::ofstream file(file_path);

                // Check if the file is open
                // std::cout << "Result" << result_vector.size() << std::endl;
                if (!file.is_open())
                {
                    std::cerr << "Failed to open file: " << file_path << std::endl;
                    return;
                }

                // Write the header
                file << "ID,distance\n";

                // Write the data
                int index_counter = 0;

                for (const auto &entry : result_vector)
                {
                    file << entry.first << "," << entry.second << "\n";
                    index_counter++;
                    if (index_counter > 20)
                    {
                        break;
                    }
                }
                // std::cout << "First_record" << std::endl;
                // Close the file
                file.close();
            }
        }

        void predicted_bPlus_tree(hnswlib::labeltype &id, std::string &left_range, std::string &right_range, std::unordered_set<int> &visited_ids, std::vector<std::pair<int, float>> &result_vector, std::unordered_set<int> &visted_clusters, const void *query_data, int &query_num)
        {
            result_vector.reserve(100000);
            uint32_t value_at_index = 0;
            // Read 3 bytes into value_at_index
            memcpy(&value_at_index, mem_for_ids_clusters + (id * 3), 3);
            // Check if the second bit is set
            bool is_second_bit_set = (value_at_index & (1 << 1)) != 0;

            // Declare a variable to hold cluster ID
            uint32_t cluster_id = value_at_index >> 8;

            // Only proceed with the logic if the cluster ID is valid
            if (is_second_bit_set)
            {
                // Use a set to avoid duplicate calculations for attributes
                std::unordered_set<unsigned int> clusters = cluster_Mem_chk[id];

                for (tableint cluster_number : clusters)
                {
                    // Use a reference to mapForCMS[cluster_number] to avoid repeated lookups
                    if (visted_clusters.count(cluster_number))
                        continue;
                    visted_clusters.insert(cluster_number);
                    BLinkTree<std::string, unsigned int> *b_plus_tree = cluster_and_associated_tree[cluster_number];

                    b_plus_tree->rangeQuery(left_range, right_range, data_level0_memory_, offsetData_, size_data_per_element_, query_data, visited_ids, result_vector);
                    // std::cout<<"Result"<< getDataByInternalId(cluster_number)<<"  "<< dist_func_param_<<std::endl;
                    //  for (const auto &pair : result)
                    //  {
                    //      for (const auto &value : pair.second)
                    //      {

                    //         if (visited_ids.count(value))
                    //         {
                    //             continue;
                    //         }

                    // dist_t dist1 = fstdistfunc_(query_data, getDataByInternalId(value), dist_func_param_);

                    //         result_vector.push_back(std::make_pair(value, dist1));
                    //         visited_ids.insert(value);

                    //         // std::cout << "Value: " << value << "    "<<dist1<<"\n";
                    //         /// All Distances here.
                    //     }
                    // }

                    //   break;
                }
            }
            else
            {

                if (!visted_clusters.count(cluster_id))
                {
                    visted_clusters.insert(cluster_id);
                    BLinkTree<std::string, unsigned int> *b_plus_tree = cluster_and_associated_tree[cluster_id];

                    // char *data_level0_memory_, size_t offsetData_,size_t size_data_per_element
                    b_plus_tree->rangeQuery(left_range, right_range, data_level0_memory_, offsetData_, size_data_per_element_, query_data, visited_ids, result_vector);

                    // for (const auto &pair : result)
                    // {

                    //     for (const auto &value : pair.second)
                    //     {
                    //         // std::cout << "Value: " << value << "\n";
                    //           if (visited_ids.count(value))
                    //             {
                    //                 continue;
                    //             }
                    //         dist_t dist1 = fstdistfunc_(query_data, getDataByInternalId(value), dist_func_param_);
                    //         result_vector.push_back(std::make_pair(value, dist1));
                    //         visited_ids.insert(value);
                    //     }
                    // }
                }
            }
        }
        // Ground truth computer
        void ground_truth_computer_for_multiattribute(const void *query_data, size_t k, std::string &start_range, std::string &end_range, std::vector<char *> &point_predicate, int &query_num)
        {
            std::vector<std::pair<int, float>> ground_truth_for_queries;
            // cout<<"Data "<< meta_data_multidiemesional_query.size()<<endl;

            for (int j = 0; j < max_elements_; j++)
            {
                char *currObj1 = getDataByInternalId(j);

                dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                bool flag = false;

                // Assuming meta_data_predicates is accessible here and contains pairs of (string, vector<char*>)
                std::pair<std::string, std::vector<char *>> pair_of_vector = meta_data_multidiemesional_query[j];

                // Check each query_predicate against each predicate in pair_of_vector.second
                for (auto &query_predicate : point_predicate)
                {
                    for (auto &predicate : pair_of_vector.second)
                    { // Fixed to use .second without ()

                        if (std::strcmp(query_predicate, predicate) == 0)
                        {
                            //  cout<<"Left"<<query_predicate<<"Right"<<predicate<<endl;
                            flag = true; // Set flag to true when a match is found
                            break;
                        }
                    }
                    if (flag)
                        break; // Exit the outer loop if a match is found
                }

                // Check if the date falls within the range and if a match was found
                if (flag)
                {
                    if (pair_of_vector.first >= start_range && pair_of_vector.first <= end_range)
                    {

                        ground_truth_for_queries.emplace_back(j, dist1);
                    }
                }
            }

            // Sort the results based on the distance (second element in pair)
            std::sort(ground_truth_for_queries.begin(), ground_truth_for_queries.end(),
                      [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                      {
                          return a.second < b.second; // Sort by float in ascending order
                      });

            // Save results to CSV
            save_to_csv(ground_truth_for_queries, "/data4/hnsw/TripClick/GroundTruth_MultiDiemension/Q" + std::to_string(query_num) + ".csv");
        }

        void clustering_multidiemensional_range(tableint sizeOfCluster)
        {

            unsigned int clusterNumber = 1;
            // For CMS code:
            unsigned int cms_counter = 0;
            std::vector<CountMinSketchMinHash> cms;
            cms.push_back(CountMinSketchMinHash());

            // Here vector tree that insert the vector data for CDDF
            std::vector<std::pair<tableint, std::string>> predicate_data_CDF;
            int counterForFilter = 0; // also for updating the cluster

            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;

            for (tableint id = 0; id < meta_data_multidiemesional_query.size(); id++)
            {
                if (visitedIds.find(id) != visitedIds.end())
                    continue;
                visitedIds.insert(id);
                // Update the Count min Sketch:
                for (const auto &predicate : meta_data_multidiemesional_query[id].second)
                {

                    cms[cms_counter].update(predicate, id, 1);
                }

                // Insert data for CDF computation here.
                predicate_data_CDF.emplace_back(id, meta_data_multidiemesional_query[id].first);
                // Do addition for Count_min_sketch.

                bit_manipulation_short(id, clusterNumber);

                counterForFilter++;

                int *data = (int *)get_linklist0(id);
                if (!data)
                    continue; // Error handling

                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);
                    // Update the CMS
                    for (const auto &predicate : meta_data_multidiemesional_query[candidateId].second)
                    {
                        cms[cms_counter].update(predicate, candidateId, 1);
                    }
                    // Inserting the CDF Value here
                    predicate_data_CDF.emplace_back(candidateId, meta_data_multidiemesional_query[candidateId].first);
                    // To Do insert Count min sketch Logic
                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                }

                // Two hop insertion
                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    if (!twoHopData)
                        continue; // Error handling

                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        visitedIds.insert(candidateIdTwoHop);
                        // Update the CMS:
                        for (const auto &predicate : meta_data_multidiemesional_query[candidateIdTwoHop].second)
                        {

                            cms[cms_counter].update(predicate, candidateIdTwoHop, 1);
                        }

                        // Insert the tree logic
                        predicate_data_CDF.emplace_back(candidateIdTwoHop, meta_data_multidiemesional_query[candidateIdTwoHop].first);
                        // Todo the CMS value

                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }

                // Updating the data structure after insertion
                if (counterForFilter >= sizeOfCluster)
                {
                    std::sort(predicate_data_CDF.begin(), predicate_data_CDF.end(),
                              [](const std::pair<tableint, std::string> &a, const std::pair<tableint, std::string> &b)
                              {
                                  return a.second < b.second;
                              });

                    // Computing the CDF
                    // Step 1: Calculate the total number of data points
                    // Step 1: Calculate the total number of data points
                    size_t n = predicate_data_CDF.size();

                    // Counting the occurance of unique values and populate the vector

                    std::map<std::string, std::vector<tableint>> count_map;
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
                    std::vector<std::pair<std::string, double>> cdf; // (value, cumulative probability)

                    // Step 4: Compute the CDF
                    double cumulative_count = 0;
                    // Create a Map here that where we insert the ID and the set that contain the range identifier and
                    // set that contain it
                    std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;
                    // Here Count is the complete vector of IDs
                    for (const auto &[value, count] : count_map)
                    {
                        cumulative_count += count.size(); // Increment cumulative count

                        double cumulative_probability = cumulative_count / n; // Calculate cumulative probability
                        // Checking the probablity and range for K MinWise hashing
                        int label = check_range_search(cumulative_probability);
                        // int labelTesting=check_range_search(cumulative_probability);
                        // cout<<"Label"<<label<<"==="<<labelTesting<<"  "<< cumulative_probability<<endl;
                        computing_and_inserting_relevant_range(label, count, map_cdf_range_k_minwise);

                        cdf.push_back({value, cumulative_probability});
                    }

                    for (const auto &pair : map_cdf_range_k_minwise)
                    {
                        int key = pair.first;
                        size_t set_size = pair.second.size();
                    }

                    /// Compute the Regression Model using Eign
                    // Insert it into the  map insert it into the map[id, Model]
                    //  Flush the predicate Vector

                    RegressionModel reg_model;
                    reg_model.train(cdf);
                    reg_model.setMapCdfRangeKMinwise(map_cdf_range_k_minwise);
                    map_for_hybrid_range_queries[clusterNumber] = std::make_pair(cms[cms_counter], reg_model);
                    cms.push_back(CountMinSketchMinHash());
                    cms_counter++;
                    clusterNumber++;
                    //  cout<<"Cluster_num"<<clusterNumber<<endl;
                    //  cms[clusterNumber-1]= CountMinSketchMinHash();
                    counterForFilter = 0;
                    predicate_data_CDF = std::vector<std::pair<tableint, std::string>>();
                    cdf = std::vector<std::pair<std::string, double>>();

                    /// Map for entering into the

                    // cms_init(&cms, count_min_width, count_min_height);         // Reinitialize the CMS
                    // bloom_filter_init(&bloom_filter, bloom_filter_size, 0.05); //  bloom_filter.clear();                                // Clear the Bloom filter (if applicable)
                }
            }
            // All  remaining insertion
            if (counterForFilter > 0)
            {
                std::sort(predicate_data_CDF.begin(), predicate_data_CDF.end(),
                          [](const std::pair<tableint, std::string> &a, const std::pair<tableint, std::string> &b)
                          {
                              return a.second < b.second;
                          });
                // Computing the CDF
                // Step 1: Calculate the total number of data points
                // Step 1: Calculate the total number of data points
                size_t n = predicate_data_CDF.size();

                // Step 2: Count occurrences of each unique value// this computation is only for CDF computation
                std::map<std::string, std::vector<tableint>> count_map;
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
                std::vector<std::pair<std::string, double>> cdf; // (value, cumulative probability)

                // Step 4: Compute the CDF
                double cumulative_count = 0;
                //  here count is the size of ids vector
                std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;
                for (const auto &[value, count] : count_map)
                {
                    cumulative_count += count.size(); // Increment cumulative count

                    double cumulative_probability = cumulative_count / n; // Calculate cumulative probability
                    int label = check_range_search(cumulative_probability);

                    computing_and_inserting_relevant_range(label, count, map_cdf_range_k_minwise);

                    cdf.push_back({value, cumulative_probability});
                }
                /// Compute the Regression Model using Eign
                // Insert it into the  map insert it into the map[id, Model]
                //  Flush the predicate Vector
                // Also in the map also insert the CMS here.

                RegressionModel reg_model;
                reg_model.train(cdf);
                reg_model.setMapCdfRangeKMinwise(map_cdf_range_k_minwise);
                map_for_hybrid_range_queries[clusterNumber] = std::make_pair(cms[cms_counter], reg_model);
                cms.push_back(CountMinSketchMinHash());
                cms_counter++;

                clusterNumber++;
                //  cms[clusterNumber-1]= CountMinSketchMinHash();
                counterForFilter = 0;
                // std::vector<std::pair<tableint, std::string>> predicate_data;
                predicate_data_CDF = std::vector<std::pair<tableint, std::string>>();
                cdf = std::vector<std::pair<std::string, double>>();
            }
        }

        void computing_and_inserting_relevant_range(int &range_no, const std::vector<tableint> &ids, std::unordered_map<int, std::set<char>> &map_cdf_range_k_minwise)
        {

            // Check if map already contains the range_no key
            auto it = map_cdf_range_k_minwise.find(range_no);
            // If the range_no is found in the map, update its set
            if (it != map_cdf_range_k_minwise.end())
            {
                // If range_no exists, we update the existing set (for example, insert elements from 'ids' or some other logic)
                for (auto id : ids)
                {
                    uint64_t hash_value_of_data = MurmurHash64B(static_cast<const void *>(&id), 25, SEED);
                    uint64_t least_significant_16_bits = hash_value_of_data & 0xFF;

                    // Convert to short and ccndition of confining its limit
                    char least_significant_bits = static_cast<char>(least_significant_16_bits);
                    // cout<<"Min Hash Size"<< min_hash_RBT[j][hashval].size()<<endl;
                    it->second.insert(static_cast<char>(least_significant_bits));
                    // it->second.insert(static_cast<short>(id));
                    int size_min_hash = it->second.size();
                    if (size_min_hash > SIZE_Of_K_Min_WISE_HASH)
                    {
                        auto item = std::prev(it->second.end()); // Get iterator to last element it is sorted so the last one has greater hash value for 8 bits
                                                                 // Remove the last element
                        it->second.erase(item);
                        // exit(0);
                    }
                }
            }
            // If map has not the range
            else
            {
                // If range_no is not found, create a new set and insert it
                std::set<char> new_set;

                // Insert values for the new range_no (using 'ids' or any logic you have)
                for (auto id : ids)
                {

                    uint64_t hash_value_of_data = MurmurHash64B(static_cast<const void *>(&id), 25, SEED);
                    uint64_t least_significant_16_bits = hash_value_of_data & 0xFF;

                    // Convert to short and ccndition of confining its limit
                    char least_significant_bits = static_cast<char>(least_significant_16_bits);
                    // cout<<"Min Hash Size"<< min_hash_RBT[j][hashval].size()<<endl;
                    new_set.insert(static_cast<char>(least_significant_bits));
                    // new_set.insert(static_cast<short>(id));

                    int size_min_hash = new_set.size();
                    if (size_min_hash > SIZE_Of_K_Min_WISE_HASH)
                    {
                        auto item = std::prev(new_set.end()); // Get iterator to last element it is sorted so the last one has greater hash value for 8 bits
                                                              // Remove the last element
                        new_set.erase(item);
                        // exit(0);
                    }
                }

                // Now insert the new range and its associated set into the map
                map_cdf_range_k_minwise[range_no] = new_set;
            }
        }

        void search_hybrid_range(const void *query_data, size_t k, std::string &start_range, std::string &end_range,
                                 std::vector<char *> &additionalData, const int &query_num, const int &efs)
        {
            std::unordered_set<int> visitedNodes;
            std::vector<std::pair<int, float>> result_vector;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = searchKnn(query_data, k, nullptr, start_range,
                                                                                         end_range, additionalData);
            std::unordered_set<int> visted_Clusters;
            std::unordered_map<unsigned int, int> map;
            // Increment counter
            // nullptr, start_range, end_range,additionalData
            result_counter_distance_computation = 0;
            // double left_range = static_cast<double>(convertDateStringToTimestamp(start_range));
            // double right_range = static_cast<double>(convertDateStringToTimestamp(end_range));
            // cout<<"Res"<<result.size()<<endl;
            // std::cout<<"Res"<<result.size()<<std::endl;
            while (!result.empty())
            {
                // Access the top element (a pair of float and labeltype)
                std::pair<float, hnswlib::labeltype> top_element = result.top();

                // Get the labeltype from the pair
                hnswlib::labeltype label = top_element.second;
                // std::cout<<label<<"Label"<<std::endl;

                if (multidiemensional_search(label, start_range, end_range, additionalData))
                {
                    // The item
                    result_vector.push_back(std::make_pair(label, top_element.first));

                    // std::string pred_str(meta_data_predicates[label][0]);
                }

                // std::pair<int, int> intersection_range = compute_popularity_intersection(label, additionalData, start_range, end_range, query_num, map, left_range, right_range);
                // //    if(intersection_range.first<256)
                // //     cout<< "intersection_range"<<intersection_range.first<<endl;

                // if ((intersection_range.first > 250))

                // {
                // //     //         map[intersection_range.second]= intersection_range.second;
                //    two_hop_search_multidiemension(query_data, additionalData, start_range, end_range, label, visitedNodes, result_vector);
                // }

                // else
                // {
                //     one_hop_search_multidiemension(query_data, additionalData, start_range, end_range, label, visitedNodes, result_vector);
                // }

                // // CDF Approach
                // //  double CDF_Difference = predictedCDF(label, start_range, end_range, visted_Clusters);

                // // // //////// one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);

                //  if ((intersection_range.first>1)&&(map.find(intersection_range.second) != map.end()))

                //      {
                //  two_hop_search_multidiemension(query_data, additionalData, start_range, end_range, label, visitedNodes, result_vector);

                //     }

                //     else
                //     {
                // one_hop_search_multidiemension(query_data, additionalData, start_range, end_range, label, visitedNodes, result_vector);
                //     }
                // //     // //    // one_hop_counter_range = one_hop_counter_range + 1;
                // //     //  }
                // map[intersection_range.second] = intersection_range.second;
                // if (query_num <600)
                // {
                //     two_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                //     two_hop_counter_range = two_hop_counter_range + 1;
                // }

                // else
                // {
                //     one_hop_search_range(query_data, start_range, end_range, label, visitedNodes, result_vector);
                //     one_hop_counter_range = one_hop_counter_range + 1;
                // }

                result.pop();
            }

            // Results Writing

            if (!result_vector.empty())
            {
                // cout<<"Counter,"<<query_num<<","<<result_counter_distance_computation<<endl;
                std::sort(result_vector.begin(), result_vector.end(),
                          [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                          {
                              return a.second < b.second; // Sort by the float value (distance) in ascending order
                          });

                //   one_hop_two_hop.emplace_back(one_hop_counter_range, two_hop_counter_range);
                //   for(auto & range:one_hop_two_hop){
                //    std::cout<<query_num <<"   One_hop==="<<one_hop_counter_range<<"Two_hop=== "<<two_hop_counter_range<<std::endl;
                //     break;
                //   }

                std::string file_path = "/data4/hnsw/TripClick/ResultMultiDiemension/" + std::to_string(efs) + "/Q" + std::to_string(query_num) + ".csv";

                // Open file in write mode
                std::ofstream file(file_path);

                // Check if the file is open
                if (!file.is_open())
                {
                    std::cerr << "Failed to open file: " << file_path << std::endl;
                    return;
                }

                // Write the header
                file << "ID,distance\n";

                // file << one_hop_counter_range << "," << two_hop_counter_range << "\n";

                // Write the data
                for (const auto &entry : result_vector)
                {
                    file << entry.first << "," << entry.second << "\n";
                }
                // std::cout << "First_record" << std::endl;
                // Close the file
                file.close();
            }
        }

        // Two hop search for multi diemensional search
        void two_hop_search_multidiemension(const void *query_data, std::vector<char *> &additionalData, std::string &start_range,
                                            std::string &end_range, hnswlib::labeltype &node,
                                            std::unordered_set<int> &visitedNodes, std::vector<std::pair<int, float>> &result_vector)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            visited_array[node] = visited_array_tag;
            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                if (visited_array[candidate_id] == visited_array_tag || !(visitedNodes.count(candidate_id) == 0))
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                visitedNodes.insert(candidate_id);

                if (multidiemensional_search(candidate_id, start_range, end_range, additionalData))
                {

                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                    result_counter_distance_computation = result_counter_distance_computation + 1;

                    result_vector.push_back(std::make_pair(candidate_id, dist1));
                }
                // Two hop searching
                int *twoHopData = (int *)get_linklist0(candidate_id);
                if (!twoHopData)
                    continue; // Error handling

                size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    if (visited_array[candidateIdTwoHop] == visited_array_tag || !(visitedNodes.count(candidateIdTwoHop) == 0))
                        continue;
                    visited_array[candidateIdTwoHop] = visited_array_tag;
                    visitedNodes.insert(candidateIdTwoHop);
                    if (multidiemensional_search(candidateIdTwoHop, start_range, end_range, additionalData))
                    {

                        char *currObj1 = (getDataByInternalId(candidateIdTwoHop));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                        result_counter_distance_computation = result_counter_distance_computation + 1;

                        result_vector.push_back(std::make_pair(candidateIdTwoHop, dist1));
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }
        // One hop Search Multiiemensional
        void one_hop_search_multidiemension(const void *query_data, std::vector<char *> &additionalData, std::string &start_range,
                                            std::string &end_range, hnswlib::labeltype &node,
                                            std::unordered_set<int> &visitedNodes, std::vector<std::pair<int, float>> &result_vector)
        {

            // void one_hop_search(const void *query_data, std::vector<char *> &additionalData, hnswlib::labeltype &node,
            //                     std::unordered_set<int> &visitedNodes)
            // {

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            visited_array[node] = visited_array_tag;
            // One Hop Neighbour
            // Data and its pointer
            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                if (visited_array[candidate_id] == visited_array_tag || !(visitedNodes.count(candidate_id) == 0))
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                visitedNodes.insert(candidate_id);
                if (multidiemensional_search(candidate_id, start_range, end_range, additionalData))
                {

                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    result_vector.push_back(std::make_pair(candidate_id, dist1));
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }

        std::pair<int, int> compute_popularity_intersection(hnswlib::labeltype &id, std::vector<char *> &metaData, std::string start_range, std::string end_range, const int &querynum,
                                                            std::unordered_map<unsigned int, int> &map, double &left_range, double &right_range)
        {
            int maximum_intersection_size = -1;
            uint32_t value_at_index = 0;

            // Read 3 bytes into value_at_index
            memcpy(&value_at_index, mem_for_ids_clusters + (id * 3), 3);

            // Check if the second bit is set
            bool is_second_bit_set = (value_at_index & (1 << 1)) != 0;

            // Declare a variable to hold cluster ID
            uint32_t cluster_id = value_at_index >> 8;

            // Random generator setup
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_int_distribution<> dis(0, 1);

            unsigned int cluster_number;
            if (is_second_bit_set)
            {
                cluster_number = getRandomElement(cluster_Mem_chk[id]);

                // Check if the cluster exists in the map
                auto it = map.find(cluster_number);
                if (it != map.end())
                {
                    return make_pair(it->second, cluster_number); // Return cached maximum intersection size
                }
            }
            else
            {
                cluster_number = cluster_id;
            }

            // Reference to the cms_instance for the selected cluster
            auto &cms_instance = map_for_hybrid_range_queries[cluster_number];

            // Predict start and end range using cms_instance
            // double left_range = static_cast<double>(convertDateStringToTimestamp(start_range));
            // double right_range = static_cast<double>(convertDateStringToTimestamp(end_range));
            // Must correct While Experiment
            int start = check_range_search(cms_instance.second.predict(left_range));
            int end = check_range_search(cms_instance.second.predict(right_range));

            if (start >= 0 && end > 0)
            {
                for (const auto &attribute : metaData)
                {
                    int key_to_use = dis(gen) == 0 ? start : end;

                    // Estimate the pair once
                    auto pair_ = cms_instance.first.estimate(attribute);
                    auto &set_difference_CMS = cms_instance.first.min_hash_RBT[pair_.first][pair_.second];

                    // Find the map entry once
                    auto map_it = cms_instance.second.getMapCdfRangeKMinwise().find(key_to_use);
                    if (map_it != cms_instance.second.getMapCdfRangeKMinwise().end())
                    {
                        int total_intersection_size = computing_intersection(set_difference_CMS, map_it->second);

                        if (total_intersection_size > maximum_intersection_size)
                        {
                            maximum_intersection_size = total_intersection_size;
                        }
                    }
                    break;
                }
            }

            // if(maximum_intersection_size<0)
            //     maximum_intersection_size=256;
            return std::make_pair(maximum_intersection_size, cluster_number);
        }
        // Search method multiiemensional query range
        bool multidiemensional_search(unsigned int identified_index, std::string &q_start_range, std::string &q_end_range, std::vector<char *> &additionalData) const
        {
            // Retrieve the pair for the identified index .at(index)
            const auto &pair_of_vector = meta_data_multidiemesional_query.at(identified_index);

            bool flag = false;

            // Check each query_predicate against each predicate in pair_of_vector.second
            for (auto &query_predicate : additionalData)
            {
                for (auto &predicate : pair_of_vector.second)
                { // Fixed to use .second without ()

                    if (std::strcmp(query_predicate, predicate) == 0)
                    {

                        flag = true; // Set flag to true when a match is found
                        break;
                    }
                }
                if (flag)
                    break; // Exit the outer loop if a match is found
            }

            // Check if the date falls within the range and if a match was found
            if (flag)
            {
                if (pair_of_vector.first >= q_start_range && pair_of_vector.first <= q_end_range)
                {
                    //  cout<<"Correct"<<endl;
                    return true;
                }
            }
            return false;
        }
        // Function to parallelize the code
        template <class Function>
        void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
        {
            if (numThreads <= 0)
            {
                numThreads = std::thread::hardware_concurrency();
            }

            if (numThreads == 1)
            {
                for (size_t id = start; id < end; id++)
                {
                    fn(id, 0);
                }
            }
            else
            {
                std::vector<std::thread> threads;
                std::atomic<size_t> current(start);

                // keep track of exceptions in threads
                // https://stackoverflow.com/a/32428427/1713196
                std::exception_ptr lastException = nullptr;
                std::mutex lastExceptMutex;

                for (size_t threadId = 0; threadId < numThreads; ++threadId)
                {
                    threads.push_back(std::thread([&, threadId]
                                                  {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                } }));
                }
                for (auto &thread : threads)
                {
                    thread.join();
                }
                if (lastException)
                {
                    std::rethrow_exception(lastException);
                }
            }
        }

        // Code For Randomly select the cluster from the cluster set for multi-diemenesional search
        unsigned int getRandomElement(const std::unordered_set<unsigned int> &clusters)
        {

            static std::vector<unsigned int> clusterVec(clusters.begin(), clusters.end());
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, clusterVec.size() - 1);

            return clusterVec[dis(gen)];

            // static std::random_device rd;                                // Random device (seeded once)
            // static std::mt19937 gen(rd());                               // Mersenne Twister PRNG
            // std::uniform_int_distribution<> dis(0, clusters.size() - 1); // Distribution for set size

            // auto it = clusters.begin();
            // std::advance(it, dis(gen)); // Move iterator to random position
            // return *it;

            // Seed for random number generation
            // std::srand(static_cast<unsigned int>(std::time(nullptr)));

            // // Generate a random index
            // auto randomIndex = std::rand() % clusters.size();

            // // Advance the iterator to the random position
            // auto it = clusters.begin();
            // std::advance(it, randomIndex);

            // return *it; // Dereference iterator to get the random element
        }
        // Computing the intersection for
        int computing_intersection(const std::set<char> &count_min_sketch_ids, const std::set<char> &regression_models)
        {

            std::set<char> intersectionSet;

            // Find the intersection of set1 and set2
            std::set_intersection(count_min_sketch_ids.begin(), count_min_sketch_ids.end(),
                                  regression_models.begin(), regression_models.end(),
                                  std::inserter(intersectionSet, intersectionSet.begin()));
            return intersectionSet.size();
            // if(intersectionSet.size()<256)
            // cout<<intersectionSet.size()<<endl;
            //  Use std::for_each to iterate over cms with a custom lambda
            //     std::for_each(count_min_sketch_ids.begin(), count_min_sketch_ids.end(), [&](int value)
            //                   {
            // if (regression_models.find(value) != regression_models.end()) {  // Check if value is in regressionModel
            //     //std::lock_guard<std::mutex> lock(mtx);  // Lock insertion to intersectionSet
            //     result_set.insert(value);
            // } });
        }
        void ProcessAttributes(
            CountMinSketchMinHash &cms_instance,
            const std::vector<char *> meta_data,
            int &maximum_frequency, std::vector<std::pair<int, std::set<char> *>> &setIntersection, const unsigned int &cluster_num, unsigned int &selected_cluster) const
        {
            for (const auto &attribute : meta_data)
            {
                const auto pair_ = cms_instance.estimate(attribute);
                const int res = cms_instance.C[pair_.first][pair_.second];
                // maximum_frequency = std::max(maximum_frequency, res);

                if (res > maximum_frequency)
                {
                    maximum_frequency = res;
                    selected_cluster = cluster_num;
                    setIntersection.push_back(std::make_pair(maximum_frequency, &cms_instance.min_hash_RBT[pair_.first][pair_.second]));
                }
            }
        }

        void ground_truth_point_predicate(const void *query_data, size_t k, std::vector<char *> &query_predicates, int &query_num)
        {

            std::vector<std::pair<int, float>> ground_truth_for_queries;
            int counterF = 0;
            // cout<<"max_elements_"<<max_elements_<<endl;
            for (hnswlib::tableint j = 0; j < max_elements_; j++)
            {
                bool res_ = result_computer_check(query_predicates, j);
                if (res_ == true)
                {
                    char *currObj1 = (getDataByInternalId(j));

                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                    // if(dist1<1.0){

                    //     for (size_t i = 0; i < query_predicates.size(); ++i) {
                    //         std::cout << query_predicates[i];
                    //         if (i < query_predicates.size() - 1) {
                    //             std::cout << ", "; // Add a separator for all except the last element
                    //         }
                    //     }
                    //     cout<<"\n";
                    //       cout<<j;
                    //       for (auto &predicate : meta_data_predicates.at(j))
                    //         {
                    //                 std::cout << predicate;
                    //         }

                    // }

                    ground_truth_for_queries.emplace_back(j, dist1);
                    counterF++;
                }
            }
            // cout<<"counter"<<counterF<<endl;

            std::sort(ground_truth_for_queries.begin(), ground_truth_for_queries.end(),
                      [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                      {
                          return a.second < b.second; // Sort by float in ascending order
                      });

            save_to_csv(ground_truth_for_queries, "/data4/hnsw/TripClick/GroundTruth/Q" + std::to_string(query_num) + ".csv");
            // save_to_csv(ground_truth_for_queries, "/data4/hnsw/tripclick_full/Ground_Truth_U/Q" + std::to_string(query_num) + ".csv");
        }

        void two_hop_search_ACORN(const void *query_data, tableint node, VisitedList *vl, std::unordered_set<tableint> &visitedNodes, std::vector<char *> additionalData, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates_ACORN) const
        {
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                //      if (!(visited_array[candidate_id] == visited_array_tag ) &&(visitedNodes.count(candidate_id) == 0)){
                //    visited_array[candidate_id] = visited_array_tag;

                //  if (visited_array[candidate_id] != visited_array_tag ){
                if ((visitedNodes.count(candidate_id) == 0))
                {
                    // continue;
                    visitedNodes.insert(candidate_id);

                    if (result_computer_check(additionalData, candidate_id))
                    {
                        char *currObj1 = (getDataByInternalId(candidate_id));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                        top_candidates_ACORN.emplace(dist1, candidate_id);
                    }
                }
                // Two hop searching

                int *twoHopData = (int *)get_linklist0(candidate_id);
                size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    if (((visitedNodes.count(candidateIdTwoHop) != 0)))
                        continue;
                    //  if (visited_array[candidate_id] == visited_array_tag )
                    //      continue;
                    // visited_array[candidateIdTwoHop] = visited_array_tag;
                    visitedNodes.insert(candidateIdTwoHop);
                    if (result_computer_check(additionalData, candidateIdTwoHop))
                    {
                        char *currObj1 = (getDataByInternalId(candidateIdTwoHop));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                        top_candidates_ACORN.emplace(dist1, candidateIdTwoHop);
                    }
                }
            }
        }

        void two_hop_search_ACORN_RANGE(const void *query_data, tableint node, VisitedList *vl, std::unordered_set<tableint> &visitedNodes, unsigned int &left_range, unsigned int &right_range, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates_ACORN) const

        {

            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                //      if (!(visited_array[candidate_id] == visited_array_tag ) &&(visitedNodes.count(candidate_id) == 0)){
                //    visited_array[candidate_id] = visited_array_tag;

                //  if (visited_array[candidate_id] != visited_array_tag ){
                if ((visitedNodes.count(candidate_id) == 0))
                {
                    // continue;
                    visitedNodes.insert(candidate_id);

                    // range_result_computer_check
                    if (range_int_computer(left_range, right_range, candidate_id))
                    {
                        char *currObj1 = (getDataByInternalId(candidate_id));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                        top_candidates_ACORN.emplace(dist1, candidate_id);
                    }
                }
                // Two hop searching

                int *twoHopData = (int *)get_linklist0(candidate_id);
                size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    if (((visitedNodes.count(candidateIdTwoHop) != 0)))
                        continue;
                    //  if (visited_array[candidate_id] == visited_array_tag )
                    //      continue;
                    // visited_array[candidateIdTwoHop] = visited_array_tag;
                    visitedNodes.insert(candidateIdTwoHop);
                    if (range_int_computer(left_range, right_range, candidateIdTwoHop))
                    {

                        char *currObj1 = (getDataByInternalId(candidateIdTwoHop));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                        top_candidates_ACORN.emplace(dist1, candidateIdTwoHop);
                    }
                }
            }
        }

        void one_hop_search_ACORN(const void *query_data, tableint node, VisitedList *vl, std::unordered_set<tableint> &visitedNodes, std::vector<char *> additionalData, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates_ACORN) const
        {
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                //      if (!(visited_array[candidate_id] == visited_array_tag ) &&(visitedNodes.count(candidate_id) == 0)){
                //    visited_array[candidate_id] = visited_array_tag;

                //  if (visited_array[candidate_id] != visited_array_tag ){
                if ((visitedNodes.count(candidate_id) == 0))
                {
                    // continue;
                    visitedNodes.insert(candidate_id);

                    if (result_computer_check(additionalData, candidate_id))
                    {
                        char *currObj1 = (getDataByInternalId(candidate_id));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                        top_candidates_ACORN.emplace(dist1, candidate_id);
                    }
                }
            }
        }

        time_t convertDateStringToTimestamp(const std::string &dateStr)
        {
            std::tm tm = {};
            std::istringstream ss(dateStr);

            // Assuming the date format is YYYY-MM-DD

            ss >> std::get_time(&tm, "%Y-%m-%d");
            if (ss.fail())
            {
                cout << "Date is" << dateStr << endl;
                throw std::invalid_argument("Date format is incorrect.");
            }

            // Convert to time_t
            return std::mktime(&tm);
        }

        bool range_int_computer(unsigned int left_range, unsigned int right_range, unsigned int index) const
        {

            unsigned int predicate = meta_data_int_.at(index);

            // cout<< "predi"<< predicate<< endl;
            if (predicate >= left_range && predicate <= right_range)
            {
                return true;
            }
            return false; // Fallback return for cases where the condition is not met
        }

        void clustering_for_cdf_range_filtering_int(tableint sizeOfCluster)
        {

            unsigned int clusterNumber = 1;
            // Here vector tree that insert the vector data for CDDF
            std::vector<std::pair<tableint, unsigned int>> predicate_data_CDF;
            int counterForFilter = 0; // also for updating the cluster

            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;
            // meta_data_multi_diemensional_qery here used as meta data.

            for (tableint id = 0; id < meta_data_int_.size(); id++)
            {

                if (visitedIds.find(id) != visitedIds.end())
                    continue;
                visitedIds.insert(id);
                // Insert data for CDF computation here.
                ////////  predicate_data_CDF.emplace_back(id, meta_data_multidiemesional_query[id].first);
                // Do addition for Count_min_sketch.

                bit_manipulation_short(id, clusterNumber);

                counterForFilter++;

                int *data = (int *)get_linklist0(id);
                if (!data)
                    continue; // Error handling

                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);

                    // Inserting the CDF Value here
                    predicate_data_CDF.emplace_back(candidateId, meta_data_int_[candidateId]);

                    // To Do insert Count min sketch Logic
                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                }

                // Two hop insertion
                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    if (!twoHopData)
                        continue; // Error handling

                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        visitedIds.insert(candidateIdTwoHop);
                        // Insert the tree logic
                        predicate_data_CDF.emplace_back(candidateIdTwoHop, meta_data_int_[candidateIdTwoHop]);
                        // Todo the CMS value

                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }

                // Updating the data structure after insertion
                if (counterForFilter >= sizeOfCluster)
                {

                    std::sort(predicate_data_CDF.begin(), predicate_data_CDF.end(),
                              [](const std::pair<tableint, unsigned int> &a, const std::pair<tableint, unsigned int> &b)
                              {
                                  return a.second < b.second;
                              });

                    // Computing the CDF
                    // Step 1: Calculate the total number of data points
                    // Step 1: Calculate the total number of data points

                    size_t n = predicate_data_CDF.size();

                    // Counting the occurance of unique values and populate the vector

                    std::map<unsigned int, std::vector<tableint>> count_map;
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
                    std::vector<std::pair<unsigned int, double>> cdf; // (value, cumulative probability)

                    // Step 4: Compute the CDF
                    double cumulative_count = 0;
                    // Create a Map here that where we insert the ID and the set that contain the range identifier and
                    // set that contain it
                    std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;
                    // Here Count is the complete vector of IDs
                    for (const auto &[value, count] : count_map)
                    {
                        cumulative_count += count.size(); // Increment cumulative count

                        double cumulative_probability = cumulative_count / n; // Calculate cumulative probability
                        // Checking the probablity and range for K MinWise hashing
                        int label = check_range_search(cumulative_probability);
                        //  int labelTesting=check_range_search(cumulative_probability);
                        //  cout<<"Label"<<label<<"==="<<labelTesting<<"  "<< cumulative_probability<<endl;
                        computing_and_inserting_relevant_range(label, count, map_cdf_range_k_minwise);

                        cdf.push_back({value, cumulative_probability});
                    }

                    /// Compute the Regression Model using Eign
                    // Insert it into the  map insert it into the map[id, Model]
                    //  Flush the predicate Vector

                    RegressionModel reg_model;

                    reg_model.train_int(cdf);
                    reg_model.setMapCdfRangeKMinwise(map_cdf_range_k_minwise);
                    mapForRegressionModel[clusterNumber] = std::move(reg_model);

                    clusterNumber++;
                    counterForFilter = 0;
                    predicate_data_CDF = std::vector<std::pair<tableint, unsigned int>>();
                    cdf = std::vector<std::pair<unsigned int, double>>();

                    /// Map for entering into the

                    // cms_init(&cms, count_min_width, count_min_height);         // Reinitialize the CMS
                    // bloom_filter_init(&bloom_filter, bloom_filter_size, 0.05); //  bloom_filter.clear();                                // Clear the Bloom filter (if applicable)
                }
            }
            // All  remaining insertion
            if (counterForFilter > 0)
            {
                std::sort(predicate_data_CDF.begin(), predicate_data_CDF.end(),
                          [](const std::pair<tableint, unsigned int> &a, const std::pair<tableint, unsigned int> &b)
                          {
                              return a.second < b.second;
                          });
                // Computing the CDF
                // Step 1: Calculate the total number of data points
                // Step 1: Calculate the total number of data points
                size_t n = predicate_data_CDF.size();

                // Step 2: Count occurrences of each unique value// this computation is only for CDF computation
                std::map<unsigned int, std::vector<tableint>> count_map;
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
                std::vector<std::pair<unsigned int, double>> cdf; // (value, cumulative probability)

                // Step 4: Compute the CDF
                double cumulative_count = 0;
                //  here count is the size of ids vector
                std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;
                for (const auto &[value, count] : count_map)
                {
                    cumulative_count += count.size(); // Increment cumulative count

                    double cumulative_probability = cumulative_count / static_cast<double>(n);
                    ; // Calculate cumulative probability

                    int label = check_range_search(cumulative_probability);

                    computing_and_inserting_relevant_range(label, count, map_cdf_range_k_minwise);

                    cdf.push_back({value, cumulative_probability});
                }
                /// Compute the Regression Model using Eign
                // Insert it into the  map insert it into the map[id, Model]
                //  Flush the predicate Vector
                // Also in the map also insert the CMS here.

                RegressionModel reg_model;
                reg_model.train_int(cdf);
                reg_model.setMapCdfRangeKMinwise(map_cdf_range_k_minwise);
                mapForRegressionModel[clusterNumber] = std::move(reg_model);
                clusterNumber++;
                counterForFilter = 0;
                predicate_data_CDF = std::vector<std::pair<tableint, unsigned int>>();
                cdf = std::vector<std::pair<unsigned int, double>>();
            }
        }

        void two_hop_search_multidiemension_ACORN(const void *query_data, VisitedList *vl, std::vector<char *> &additionalData,
                                                  std::string &start_range, std::string &end_range,
                                                  const hnswlib::labeltype &node, std::unordered_set<tableint> &visitedNodes,
                                                  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates_ACORN) const
        {

            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            int *data;
            data = (int *)get_linklist0(node);
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
            // Getting the nearest neighbours size
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

                //      if (!(visited_array[candidate_id] == visited_array_tag ) &&(visitedNodes.count(candidate_id) == 0)){
                //    visited_array[candidate_id] = visited_array_tag;

                //  if (visited_array[candidate_id] != visited_array_tag ){
                if ((visitedNodes.count(candidate_id) == 0))
                {
                    // continue;
                    visitedNodes.insert(candidate_id);

                    // range_result_computer_check
                    if (multidiemensional_search(candidate_id, start_range, end_range, additionalData))
                    {
                        char *currObj1 = (getDataByInternalId(candidate_id));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);

                        top_candidates_ACORN.emplace(dist1, candidate_id);
                    }
                }
                // Two hop searching

                int *twoHopData = (int *)get_linklist0(candidate_id);
                size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    if (((visitedNodes.count(candidateIdTwoHop) != 0)))
                        continue;
                    //  if (visited_array[candidate_id] == visited_array_tag )
                    //      continue;
                    // visited_array[candidateIdTwoHop] = visited_array_tag;
                    visitedNodes.insert(candidateIdTwoHop);
                    if (multidiemensional_search(candidateIdTwoHop, start_range, end_range, additionalData))
                    {

                        char *currObj1 = (getDataByInternalId(candidateIdTwoHop));

                        dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                        top_candidates_ACORN.emplace(dist1, candidateIdTwoHop);
                    }
                }
            }
        }

        void computing_data_for_finding_corelation(const void *query_data, size_t k, std::vector<char *> query_predicates, int &query_num)
        {
            std::vector<std::pair<int, float>> ground_truth_for_queries;
            for (int j = 0; j < max_elements_; j++)
            {
                char *currObj1 = (getDataByInternalId(j));
                dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                ground_truth_for_queries.emplace_back(j, dist1);
            }

            std::sort(ground_truth_for_queries.begin(), ground_truth_for_queries.end(),
                      [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                      {
                          return a.second < b.second; // Sort by float in descending order
                      });

            save_to_csv_corelation(ground_truth_for_queries, "/data3/"
                                                             "/Corelation_Graph/Q" +
                                                                 std::to_string(query_num) + ".csv",
                                   query_predicates);
        }

        void save_to_csv_corelation(const std::vector<std::pair<int, float>> &data, const std::string &filename, std::vector<char *> &query_predicates)
        {

            std::string result;

            for (size_t i = 0; i < query_predicates.size(); ++i)
            {
                result += query_predicates[i];
                if (i != query_predicates.size() - 1)
                {
                    result += ";";
                }
            }

            std::ofstream file(filename); // Open the file

            if (file.is_open())
            {
                // Write the header (optional)
                // file << "ID,Distance\n";

                // Write the data
                int c = 0;
                file << "ID,distance, predicate\n";
                for (const auto &pair : data)
                {
                    // Each pair is written as ID,Distance

                    if (c < 1000)
                    {
                        file << pair.first << "," << pair.second << "," << result << "\n";
                    }
                    else
                    {
                        file << pair.first << "," << pair.second << "\n";
                    }
                    c++;
                }

                file.close(); // Close the file
                std::cout << "Data saved to " << filename << " successfully!" << std::endl;
            }
            else
            {
                std::cerr << "Error: Unable to open file " << filename << std::endl;
            }
        }

        void clustering_for_corelation(int sizeOfCluster)
        {

            unsigned int cms_counter = 0;
            unsigned int clusterNumber = 1;
            /// Correct count Map
            std::unordered_map<std::string, unsigned int> correctMap_;
            int counterForFilter = 0; // also for updating the cluster

            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;

            for (tableint id = 0; id < meta_data_predicates.size(); id++)
            {

                if (visitedIds.find(id) != visitedIds.end())
                    continue;

                visitedIds.insert(id);
                // Insert data
                for (const auto &predicate : meta_data_predicates.at(id))
                {
                    std::string str(predicate);
                    auto it = correctMap_.find(str);
                    if (it != correctMap_.end())
                    {
                        it->second++; // Increment if found
                    }
                    else
                    {
                        correctMap_[str] = 1; // Insert if not found
                    }
                }
                corelation_id[id] = clusterNumber;
                //   bit_manipulation_short(id, clusterNumber);

                counterForFilter++;

                int *data = (int *)get_linklist0(id);

                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);

                    if (visitedIds.find(candidateId) != visitedIds.end())
                        continue;

                    visitedIds.insert(candidateId);

                    for (const auto &predicate : meta_data_predicates.at(candidateId))
                    {
                        std::string str(predicate);
                        auto it = correctMap_.find(str);
                        if (it != correctMap_.end())
                        {
                            it->second++; // Increment if found
                        }
                        else
                        {
                            correctMap_[str] = 1; // Insert if not found
                        }
                    }
                    corelation_id[candidateId] = clusterNumber;
                    // cluster_hash_based_updating(candidateId, clusterNumber);
                    // bit_manipulation(candidateId, clusterNumber);
                    // bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;
                    // }

                    // // Two hop insertion
                    // for (size_t j = 0; j < size; j++)
                    // {
                    // tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);

                        if (visitedIds.find(candidateIdTwoHop) != visitedIds.end())
                            continue;
                        visitedIds.insert(candidateIdTwoHop);

                        for (const auto &predicate : meta_data_predicates.at(candidateIdTwoHop))
                        {
                            std::string str(predicate);
                            auto it = correctMap_.find(str);
                            if (it != correctMap_.end())
                            {
                                it->second++; // Increment if found
                            }
                            else
                            {
                                correctMap_[str] = 1; // Insert if not found
                            }
                        }

                        // cluster_hash_based_updating(candidateIdTwoHop, clusterNumber);
                        corelation_id[candidateIdTwoHop] = clusterNumber;
                        // bit_manipulation(candidateIdTwoHop, clusterNumber);
                        // bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }

                // Updating the data structure after insertion
                if (counterForFilter >= sizeOfCluster)
                {
                    // cout<<"counterForFilter::: "<<counterForFilter<<endl;

                    // mapForCMS[clusterNumber].saveToFile("/data4/hnsw/paper/Clusters/CMS_256_512/" + std::to_string(clusterNumber) + ".bin");

                    //  cms[clusterNumber-1]= CountMinSketchMinHash();
                    counterForFilter = 0;
                    correctMap[clusterNumber] = correctMap_;
                    clusterNumber++;
                    correctMap_ = std::unordered_map<std::string, unsigned int>();
                }
            }
            // All  remaining insertion
            if (counterForFilter > 0)
            {

                correctMap[clusterNumber] = correctMap_;
                correctMap_ = std::unordered_map<std::string, unsigned int>();

                // Writing IDS
                //  writeIdsAndClusterRelationShip("/data4/hnsw/paper/Clusters//map_256_512.bin", "/data4/hnsw/paper/Clusters//short_256_512.bin");
            }
        }

        unsigned int compute_corelation_pop(const hnswlib::labeltype &id, std::vector<std::string> &metaData) const
        {
            const unsigned int &cluster = corelation_id.at(id);
            const std::unordered_map<std::string, unsigned int> map = correctMap.at(cluster);

            for (std::string &key_ : metaData)
            {

                auto it = map.find(key_);

                if (it != map.end()) // If the key exists in the map
                {

                    std::cout << "Key: " << key_ << " Value: " << it->second << std::endl;
                    return it->second;
                }
                else // If the key doesn't exist in the map
                {
                    return 0;
                }
            }

            // cout << "Map Size" << map.size() << endl;
            // for (const auto &pair : map)
            // {
            //     std::string key = pair.first;
            //     for (char *key_ : metaData)
            //     {
            //         std::string str(key_);

            //         if (key == str)
            //             cout << "I am here" << endl;
            //     }
            // }
            //         if (strcmp(key, key_) == 0)
            //        {
            //          auto it = map.find(key_);

            //             cout << key << " " << "key   " <<key_<<it->second <<endl;
            //         }
            //     }
            // }
            return -1;
        }

        // This

        void clustering_and_maintaining_sketch_test_memory(tableint sizeOfCluster)
        {

            std::unordered_map<tableint, std::unordered_set<unsigned int>> cluster_mpping;

            std::vector<CountMinSketchMinHash> cms;
            unsigned int cms_counter = 0;
            unsigned int clusterNumber = 1;
            cms.push_back(CountMinSketchMinHash());
            /// Correct count Map

            int counterForFilter = 0; // also for updating the cluster

            // For keeping track of visited nodes
            std::unordered_set<tableint> visitedIds;

            for (tableint id = 0; id < meta_data_predicates.size(); id++)
            {

                if (visitedIds.find(id) != visitedIds.end())
                    continue;

                visitedIds.insert(id);
                // Insert data
                for (const auto &predicate : meta_data_predicates[id])
                {

                    cms[cms_counter].update(predicate, id, 1);
                    cluster_mpping[id].insert(clusterNumber);

                    const auto pair_ = cms[cms_counter].estimate(predicate);
                    const int maximum = cms[cms_counter].C[pair_.first][pair_.second];

                    bit_array_test(cluster_mpping[id], id, clusterNumber, maximum, predicate);
                }

                counterForFilter++;

                int *data = (int *)get_linklist0(id);

                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                // One hop insertion

                for (size_t j = 0; j < size; j++)
                {

                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);

                    for (const auto &predicate : meta_data_predicates[candidateId])
                    {

                        cms[cms_counter].update(predicate, candidateId, 1);
                        if (cluster_mpping.find(candidateId) != cluster_mpping.end())
                        {
                            // If present, add the clusterNumber to the set
                            cluster_mpping[candidateId].insert(clusterNumber);
                            const auto pair_ = cms[cms_counter].estimate(predicate);
                            const int maximum = cms[cms_counter].C[pair_.first][pair_.second];
                            bit_array_test(cluster_mpping[candidateId], candidateId, clusterNumber, maximum, predicate);
                        }
                        else
                        {
                            std::unordered_set<unsigned int> newSet;
                            newSet.insert(candidateId); // First id
                            // newSet.insert(cluster_identifier_); // Second ID
                            cluster_mpping[candidateId] = newSet;
                            const auto pair_ = cms[cms_counter].estimate(predicate);
                            const int maximum = cms[cms_counter].C[pair_.first][pair_.second];
                            bit_array_test(cluster_mpping[candidateId], candidateId, clusterNumber, maximum, predicate);
                        }
                    }

                    // cluster_hash_based_updating(candidateId, clusterNumber);
                    // bit_manipulation(candidateId, clusterNumber);
                    //  bit_manipulation_short(candidateId, clusterNumber);

                    counterForFilter++;
                    // }

                    // // Two hop insertion
                    // for (size_t j = 0; j < size; j++)
                    // {
                    // tableint candidateId = *(datal + j);
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        visitedIds.insert(candidateIdTwoHop);

                        for (const auto &predicate : meta_data_predicates[candidateIdTwoHop])
                        {

                            cms[cms_counter].update(predicate, candidateIdTwoHop, 1);

                            if (cluster_mpping.find(candidateIdTwoHop) != cluster_mpping.end())
                            {
                                // If present, add the clusterNumber to the set
                                cluster_mpping[candidateIdTwoHop].insert(clusterNumber);
                                const auto pair_ = cms[cms_counter].estimate(predicate);
                                const int maximum = cms[cms_counter].C[pair_.first][pair_.second];
                                bit_array_test(cluster_mpping[candidateIdTwoHop], candidateIdTwoHop, clusterNumber, maximum, predicate);
                            }
                            else
                            {
                                std::unordered_set<unsigned int> newSet;
                                newSet.insert(candidateIdTwoHop); // First id
                                // newSet.insert(cluster_identifier_); // Second ID
                                cluster_mpping[candidateIdTwoHop] = newSet;
                                const auto pair_ = cms[cms_counter].estimate(predicate);
                                const int maximum = cms[cms_counter].C[pair_.first][pair_.second];
                                bit_array_test(cluster_mpping[candidateIdTwoHop], candidateIdTwoHop, clusterNumber, maximum, predicate);
                            }
                        }

                        // cluster_hash_based_updating(candidateIdTwoHop, clusterNumber);
                        // bit_manipulation(candidateIdTwoHop, clusterNumber);
                        // bit_manipulation_short(candidateIdTwoHop, clusterNumber);

                        counterForFilter++;
                    }
                }

                // Updating the data structure after insertion
                if (counterForFilter >= sizeOfCluster)
                {
                    // cout<<"counterForFilter::: "<<counterForFilter<<"Cluster"<<clusterNumber<<endl;

                    mapForCMS[clusterNumber] = cms[cms_counter]; // Store the CMS

                    // Writing for testing
                    // cms[cms_counter].saveToFile("/data3/""/CMS_size/" + std::to_string(clusterNumber) + ".bin");

                    // mapForCMS[clusterNumber].saveToFile("/data4/hnsw/paper/Clusters/CMS_256_512/" + std::to_string(clusterNumber) + ".bin");
                    clusterNumber++;
                    cms_counter++;
                    cms.push_back(CountMinSketchMinHash());
                    //  cms[clusterNumber-1]= CountMinSketchMinHash();
                    counterForFilter = 0;

                    // cms_init(&cms, count_min_width, count_min_height);         // Reinitialize the CMS
                    // bloom_filter_init(&bloom_filter, bloom_filter_size, 0.05); //  bloom_filter.clear();                                // Clear the Bloom filter (if applicable)
                }
            }
            // All  remaining insertion
            if (counterForFilter > 0)
            {

                // mapForBF[clusterNumber] = std::move(bloom_filter);
                mapForCMS[clusterNumber] = cms[cms_counter];
                // cms[cms_counter].saveToFile("/data3/""/CMS_size/" + std::to_string(clusterNumber) + ".bin");

                // mapForCMS[clusterNumber].saveToFile("/data4/hnsw/paper/Clusters/CMS_256_512/" + std::to_string(clusterNumber) + ".bin");

                // Writing IDS
                //  writeIdsAndClusterRelationShip("/data4/hnsw/paper/Clusters//map_256_512.bin", "/data4/hnsw/paper/Clusters//short_256_512.bin");
            }
        }

        void bit_array_test(const std::unordered_set<unsigned int> &set_, const int &node_identifier_, const unsigned int &cluster_identifier_, const int &maximum, const char *predicate)
        {
            int max_ = maximum;
            unsigned int cluster_ide = cluster_identifier_;

            const std::unordered_set<unsigned int> &value_set = set_;
            for (const auto &value : value_set)
            {
                CountMinSketchMinHash &sketch = mapForCMS[value];

                const auto pair_ = sketch.estimate(predicate);
                const int pred = sketch.C[pair_.first][pair_.second];
                if (pred > max_)
                {
                    max_ = pred;
                    cluster_ide = value;
                }
            }

            uint32_t value_at_index = 0;
            memcpy(&value_at_index, mem_for_ids_clusters + (node_identifier_ * 3), 3);
            value_at_index |= (cluster_ide & 0xFFFFFF); // Masking for safty
            memcpy(mem_for_ids_clusters + (node_identifier_ * 3), &value_at_index, 3);
        }

        int compute_popularity_CMS_Test(hnswlib::labeltype &id, std::vector<char *> &metaData)
        {
            uint32_t cluster = 0;
            memcpy(&cluster, mem_for_ids_clusters + (id * 3), 3);

            CountMinSketchMinHash &sketch = mapForCMS[cluster];

            const auto pair_ = sketch.estimate(metaData[0]);
            const int pred = sketch.C[pair_.first][pair_.second];
            return pred;
        }

        // computing the location and setting the respective on
        // Here figureprint should be 8, 16, 32, 64 bit
        void compute_location(unsigned int &id, vector<char *> &predicates, int file_size, int size_of_figureprint)
        {
            for (int i = 0; i < 3; i++) // here 3 is the number of hash funtion
            {
                int seed = 45; // Every time seed changing seed will produce different hash value
                uint32_t hash_value = MurmurHash64B(static_cast<const void *>(&id), sizeof(id), seed + i);
                hash_value = hash_value % 3000000;
                // Calculate byte  position
                size_t index = (hash_value) / size_of_figureprint; // Round off to exact position
                for (const auto &predicate : predicates)
                {
                    uint32_t hash_value_predicate_bit_position = MurmurHash64B(static_cast<const void *>(predicate), strlen(predicate), seed + i);
                    hash_value_predicate_bit_position = hash_value_predicate_bit_position % size_of_figureprint;
                    // Set the corresponding bit
                    bit_array_for_disk_access[index] |= (1 << hash_value_predicate_bit_position);
                }
            }
        }
        // It is boolean method which return if the file has corresponsding predicate . If it contain that predicate it return true otherwise false

        bool compute_file_check(const unsigned int &id, const char *predicate, int file_size, int size_of_figureprint) const
        {
            bool file_check = false; // Check condition
            for (int i = 0; i < 3; i++)
            {
                int seed = 45;
                uint32_t hash_value = MurmurHash64B(static_cast<const void *>(&id), sizeof(id), seed + i);
                hash_value = hash_value % 3000000;
                // Calculate   position

                size_t index = (hash_value) / size_of_figureprint;
                uint32_t hash_value_predicate_bit_position = MurmurHash64B(static_cast<const void *>(predicate), strlen(predicate), seed + i);
                hash_value_predicate_bit_position = hash_value_predicate_bit_position % size_of_figureprint;

                if (bit_array_for_disk_access[index] & (1 << hash_value_predicate_bit_position)) // Left shift masking with & to get the specific position is it on or OFF
                {
                    file_check = true;
                }
                else
                {
                    return false;
                }
            }
            return file_check;
        }
        /// Disk Access Optimization

        void clustering_and_maintaining_sketch_with_disk_optimization(tableint sizeOfCluster)
        {
            // Define the memory block for Disk optimization access
            unsigned int file_count = 3000000;
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
                int *data = (int *)get_linklist0(id);
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidateId = *(datal + j);
                    visitedIds.insert(candidateId);
                    compute_location(candidateId, meta_data_predicates[candidateId], file_count, figureprint_size);

                    for (const auto &predicate : meta_data_predicates[candidateId])
                    {
                        cms[cms_counter].update(predicate, candidateId, 1);
                    }

                    // Update clusters and bitmaps
                    bit_manipulation(candidateId, clusterNumber);
                    bit_manipulation_short(candidateId, clusterNumber);
                    counterForFilter++;

                    // Two-hop insertion (linked nodes of the candidate node)
                    int *twoHopData = (int *)get_linklist0(candidateId);
                    size_t twoHopSize = getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);

                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);
                        visitedIds.insert(candidateIdTwoHop);
                        compute_location(candidateIdTwoHop, meta_data_predicates[candidateIdTwoHop], file_count, figureprint_size);
                        for (const auto &predicate : meta_data_predicates[candidateIdTwoHop])
                        {
                            cms[cms_counter].update(predicate, candidateIdTwoHop, 1);
                        }

                        // Update clusters and bitmaps
                        bit_manipulation(candidateIdTwoHop, clusterNumber);
                        bit_manipulation_short(candidateIdTwoHop, clusterNumber);
                        counterForFilter++;
                    }
                }

                // After processing current cluster, check if we need to save the cluster
                if (counterForFilter >= sizeOfCluster)
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
        // Compute the disk access
        void compute()
        {
            for (int i = 0; i < 3; i++)
            {
                int id = 10;
                int seed = 45;
                uint32_t hash_value = MurmurHash64B(static_cast<const void *>(&id), sizeof(id), seed + i);
                cout << "Hash " << hash_value << endl;
            }

            for (int i = 0; i < 3; i++)
            {
                int id = 10;
                int seed = 45;
                uint32_t hash_value = MurmurHash64B(static_cast<const void *>(&id), sizeof(id), seed + i);
                cout << "Hash Second " << hash_value << endl;
            }
        }

        bool finding_disk_data_access(char *&predicate, const hnswlib::labeltype &id)
        {
            bool file_check = compute_file_check(id, predicate, 3000000, 8);
            if (!file_check)
                return file_check;

            // Construct the file path
            std::string filePath = "/data3/"
                                   "/Disk_optimization/Paper/" +
                                   std::to_string(id) + ".txt";

            // Open the file in read mode
            std::ifstream file(filePath);

            if (file.is_open())
            {
                std::string firstLine;
                if (std::getline(file, firstLine))
                {
                    file.close(); // Close the file
                    // Compare the first line with the predicate
                    return firstLine == predicate;
                }
                else
                {
                    file.close();
                    return false; // File is empty or error reading the first line
                }
            }
            else
            {
                // File does not exist or cannot be opened
                std::cerr << "File does not exist or cannot be opened: " << filePath << std::endl;
                return false; // Return false since the file cannot be checked
            }
        }
    };

} // namespace hnswlib
