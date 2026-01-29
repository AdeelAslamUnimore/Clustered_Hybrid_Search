#include "../../hnswlib/hnswlib.h"
#include "../../predictive_range_queries/predictive_range_hnsw.h"
#include <thread>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <iostream>

using namespace std;
std::unordered_map<unsigned int, int> reading_meta_data(const string &file_path);

vector<float> splitToFloat(const string &str, char delimiter)
{
    vector<float> tokens;
    string token;
    istringstream tokenStream(str);
    while (getline(tokenStream, token, delimiter))
    {
        try
        {
            tokens.push_back(stof(token));
        }
        catch (const invalid_argument &e)
        {
            // Handle the case where conversion to double fails
            cerr << "Warning: Invalid float value encountered: " << token << endl;
        }
    }
    return tokens;
}

bool isNullOrEmpty(const string &str)
{
    return str.empty() || str == "null";
}

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
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

pair<vector<vector<float>>, vector<pair<int, int>>> reading_queries(const string &file_path, int &dim);

int main(int argc, char const *argv[])
{

    if (__builtin_cpu_supports("avx2"))
    {
        std::cout << "AVX2 supported!\n";
    }
    else
    {
        std::cout << "AVX2 NOT supported!\n";
    }
    exit(-1);

    /* code */
    int dim = 128;
    std::unordered_map<unsigned int, int> meta_data = reading_meta_data("/home/aa5f25/siftsmall/siftsmall_docs.csv");
    int max_elements = meta_data.size();
    hnswlib::L2Space space(dim);
    clustered_hybrid_search::PredictiveRangeHNSW<float> *alg_query_aware = new clustered_hybrid_search::PredictiveRangeHNSW<float>(&space, max_elements, "/home/aa5f25/Index/index.bin", meta_data);
    alg_query_aware->clustering_for_cdf_range_filtering(1000);
    alg_query_aware->testing_function();

    //     pair<vector<vector<float>>, vector<pair<int, int>>> query_reading_results = reading_queries("/home/aa5f25/siftsmall/siftsmall_queries.csv", dim);
    //     float *query_data = new float[dim * query_reading_results.first.size()];
    //     int index_of_query_vector = 0;
    //     int size_of_query_items = query_reading_results.first.size();
    //     for (const auto &vec : query_reading_results.first)
    //     {

    //         // Iterate over each float in the current vector
    //         for (float value : vec)
    //         {
    //             if (index_of_query_vector < dim * size_of_query_items)
    //             { // Check to avoid out-of-bounds access
    //                 query_data[index_of_query_vector] = value;
    //                 index_of_query_vector++;
    //             }
    //             else
    //             {
    //                 std::cerr << "Error: data array out of bounds" << std::endl;
    //                 break;
    //             }
    //         }
    //     }

    //    //alg_query_aware->freeMemory();
    //     delete[] query_data;
    delete alg_query_aware;
}

// Read embeddings from CSV-like file for Point predicate
std::unordered_map<unsigned int, int> reading_meta_data(const std::string &file_path)
{
    std::cout << "meta_data " << file_path << std::endl;

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::unordered_map<unsigned int, int> meta_data;
    std::string line;

    // Skip header
    std::getline(file, line);

    unsigned int line_count = 0;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string embedding;
        std::string attribute;

        // Format: embedding;attribute;
        std::getline(ss, embedding, ';');
        std::getline(ss, attribute, ';');

        // Trim whitespace (optional but safe)
        attribute.erase(0, attribute.find_first_not_of(" \t\n\r\f\v"));
        attribute.erase(attribute.find_last_not_of(" \t\n\r\f\v") + 1);

        try
        {
            int value = std::stoi(attribute);
            meta_data[line_count] = value;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(
                "Invalid integer attribute at line " + std::to_string(line_count));
        }

        line_count++;
    }

    return meta_data;
}

pair<vector<vector<float>>, vector<pair<int, int>>>
reading_queries(const string &file_path, int &dim)
{
    cout << "Reading file: " << file_path << endl;
    ifstream file(file_path);

    if (!file.is_open())
    {
        throw runtime_error("Could not open file: " + file_path);
    }

    vector<vector<float>> total_embeddings;
    vector<pair<int, int>> all_attributes;
    string line;

    getline(file, line); // skip header

    while (getline(file, line))
    {
        stringstream ss(line);
        string embedding, left_attr, right_attr;

        getline(ss, embedding, ';');
        getline(ss, left_attr, ';');
        getline(ss, right_attr, ';');

        if (isNullOrEmpty(embedding) ||
            isNullOrEmpty(left_attr) ||
            isNullOrEmpty(right_attr))
            continue;

        vector<float> embeddingVector = splitToFloat(embedding, ',');

        if (dim == 0)
            dim = embeddingVector.size();

        if (embeddingVector.size() != dim)
            continue;

        try
        {
            total_embeddings.push_back(embeddingVector);
            all_attributes.emplace_back(stoi(left_attr), stoi(right_attr));
        }
        catch (...)
        {
            continue;
        }
    }

    return {total_embeddings, all_attributes};
}

void batch_process_queries(clustered_hybrid_search::PredictiveRangeHNSW<float> *alg_query_aware, float *query_data, std::vector<std::pair<int, int>> &range_queries_meta_data, std::unordered_map<unsigned int, int> &meta_data, int dim, size_t num_threads)
{
    size_t batch_size = 1000;
    size_t total_elements = alg_query_aware->max_elements_;
    size_t num_batches = (range_queries_meta_data.size() + batch_size - 1) / batch_size;
    int counter = 0;

    for (size_t b = 0; b < num_batches; b++)
    {

        counter++;
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, range_queries_meta_data.size());
        // create_directory_if_not_exists(constants["FILTER_PATH"]);
        // std::string filter_file = constants["FILTER_PATH"] + "/filter_batch_" + std::to_string(start) + ".bin";
        std::vector<char> filter_ids_map;

        // if (fileExists(filter_file))
        // {
        //     // Load precomputed map
        //     filter_ids_map = loadFilterMap(filter_file, (end - start) * total_elements);
        //     //  std::cout << "✅ Loaded cached filter map for batch " << b + 1 << std::endl;
        // }
        // else
        // {
        // Compute fresh
        // std::cout << "💾 Saving " << start << " to cache" << std::endl;
        filter_ids_map.resize((end - start) * total_elements);

        for (size_t i = start; i < end; i++)
        {
            // const std::string &attribute = queries_meta_data[i];
            std::pair<int, int> range_query_attributes = range_queries_meta_data[i];

            ParallelFor(0, alg_query_aware->max_elements_, num_threads, [&](size_t row, size_t threadId)
                        {
                   
                    //bool match_found = (attribute == meta_data_attributes[row]);
                    const auto &row_attributes = meta_data[row];

                        bool match_found = false;

                        if( (row_attributes >= range_query_attributes.first) && (row_attributes <= range_query_attributes.second))
                        {
                            match_found = true;
                        }

                   //  (attribute >= meta_data_attributes[row]);
                    filter_ids_map[(i - start) * total_elements + row] = match_found; });
        }

        // saveFilterMap(filter_ids_map, filter_file);
        std::cout << "💾 Saved filter map for batch " << b + 1 << " to cache" << std::endl;

        alg_query_aware->predicateCondition(filter_ids_map.data());
        alg_query_aware->setEf(10);
        // ------------------ Run queries ONLY for this batch ------------------
        auto batch_start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = start; i < end; i++)
        {
            alg_query_aware->search(
                query_data + (i * dim),
                i,
                range_queries_meta_data[i],
                50);
        }

        auto batch_end_time = std::chrono::high_resolution_clock::now();

        auto duration_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(
                batch_end_time - batch_start_time)
                .count() /
            1000.0;

        std::cout << "Batch " << b + 1
                  << " processed in " << duration_ms << " ms" << std::endl;
    }
}