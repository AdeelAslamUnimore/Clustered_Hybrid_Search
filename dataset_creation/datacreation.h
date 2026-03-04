#pragma once
#include "../hnswlib/hnswlib.h"
#include "../hnswlib/visited_list_pool.h"
#include "../predictive_range_queries/ranges.h"
#include "../predictive_point_queries/memory_access.h"
#include "../predictive_point_queries/count_min_sketch_min_hash.h"
#include "../predictive_range_queries/regression.h"

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
    class DatasetCreation : public hnswlib::HierarchicalNSW<dist_t>
    {

    public:
        size_t max_elements_;
        float popularity_threshold;
        std::unordered_map<size_t, std::pair<float, float>> cms_bounds_cache; // Cache for CMS bounds

    public:
        DatasetCreation(hnswlib::SpaceInterface<dist_t> *space, size_t max_elements, const std::string &location_of_index)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements)
        {

            this->loadIndex(location_of_index, space, max_elements);
            max_elements_ = max_elements;
        }

        void DescendingOrderDistance(const float *query_vec,
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

            std::cout << "I am here " << total_elements << std::endl;

            // Print first 5 values of the query vector
            std::cout << "First 5 values of query vector: ";
            for (size_t i = 0; i < std::min<size_t>(5, total_elements); ++i)
                std::cout << query_vec[i] << " ";
            std::cout << std::endl;

            std::priority_queue<std::pair<dist_t, size_t>> top_candidates;

            // Compute all distances
            for (size_t i = 0; i < total_elements; ++i)
            {
                const char *ep_data = this->getDataByInternalId(i);

                dist_t dist = this->fstdistfunc_(
                    query_vec,
                    ep_data,
                    this->dist_func_param_);

                top_candidates.emplace(dist, i);
            }

            std::cout << "I am here Computed" << std::endl;

            // Store (id, distance)
            std::vector<std::pair<size_t, dist_t>> results;

            while (!top_candidates.empty())
            {
                results.emplace_back(
                    top_candidates.top().second, // id
                    top_candidates.top().first   // distance
                );
                top_candidates.pop();
            }

            // std::reverse(results.begin(), results.end());

            std::string filename =
                "/scratch/aa5f25/datasets//TripClick/Corelation/Q" +
                std::to_string(query_id_global) + ".csv";

            std::ofstream out(filename);
            if (!out.is_open())
            {
                std::cerr << "❌ Cannot open file\n";
                return;
            }

            // Format: QueryID;id1:dist1,id2:dist2,...
            out << query_id_global << ";";

            size_t limit = std::min(k, results.size());
            for (size_t i = 0; i < limit; ++i)
            {
                out << results[i].first;
                if (i != limit - 1)
                    out << ",";
            }

            out << "\n";
            out.close();

            std::cout << "✅ Query " << query_id_global << " processed.\n";
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
    };
}
