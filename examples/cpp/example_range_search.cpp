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
void batch_process_queries(clustered_hybrid_search::PredictiveRangeHNSW<float> *alg_query_aware, float *query_data, std::vector<std::pair<int, int>> &range_queries_meta_data, std::unordered_map<unsigned int, int> &meta_data, int dim, size_t num_threads, std::unordered_map<std::string, std::string> &constants);
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
std::vector<int> parseIntList(const std::string &str);
std::unordered_map<std::string, std::string> reading_constants(const std::string &path);
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
// Load filter map
std::vector<char> loadFilterMap(const std::string &filename, size_t expected_size)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error(
            "❌ Cannot open file for reading: " + filename);
    }

    std::vector<char> buffer(expected_size);
    in.read(buffer.data(), expected_size);
    if (in.gcount() != static_cast<std::streamsize>(expected_size))
    {
        throw std::runtime_error("❌ Filter file size mismatch!");
    }
    in.close();
    return buffer;
}
// Save filter map for reuse
void saveFilterMap(const std::vector<char> &filter_ids_map, const std::string &filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
    {
        throw std::runtime_error(
            "❌ Cannot open file for writing: " + filename);
    }
    out.write(filter_ids_map.data(), filter_ids_map.size());
    out.close();
}
// Simple file-exists check (works everywhere)
bool fileExists(const std::string &filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

int main(int argc, char const *argv[])
{

    // if (__builtin_cpu_supports("avx2"))
    // {
    //     std::cout << "AVX2 supported!\n";
    // }
    // else
    // {
    //     std::cout << "AVX2 NOT supported!\n";
    // }

    std::string path_constants = "../examples/constants/range_query.txt";
    std::unordered_map<std::string, std::string> constants = reading_constants(path_constants);

    /* code */
    int dim = std::stoi(constants.at("DIM"));
    std::unordered_map<unsigned int, int> meta_data = reading_meta_data(constants.at("META_DATA_PATH"));
    int max_elements = meta_data.size();
    hnswlib::L2Space space(dim);

    clustered_hybrid_search::PredictiveRangeHNSW<float> *alg_query_aware = new clustered_hybrid_search::PredictiveRangeHNSW<float>(&space, max_elements, constants.at("INDEX_PATH"), meta_data);

    alg_query_aware->clustering_for_cdf_range_filtering(10000);
    // alg_query_aware->testing_function();

    pair<vector<vector<float>>, vector<pair<int, int>>> query_reading_results = reading_queries(constants.at("QUERIES_PATH"), dim);

    float *query_data = new float[dim * query_reading_results.first.size()];
    int index_of_query_vector = 0;
    int size_of_query_items = query_reading_results.first.size();

    for (const auto &vec : query_reading_results.first)
    {

        // Iterate over each float in the current vector
        for (float value : vec)
        {
            if (index_of_query_vector < dim * size_of_query_items)
            { // Check to avoid out-of-bounds access
                query_data[index_of_query_vector] = value;
                index_of_query_vector++;
            }
            else
            {
                std::cerr << "Error: data array out of bounds" << std::endl;
                break;
            }
        }
    }
    int num_threads = std::stoi(constants.at("NUM_THREADS"));
    batch_process_queries(alg_query_aware, query_data, query_reading_results.second, meta_data, dim, num_threads, constants);

    delete[] query_data;
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
        std::string embedding, skip1, skip2, skip3, skip4, skip5;
        std::string attribute;

        // Format: embedding;attribute;
        std::getline(ss, embedding, ';');
        std::getline(ss, skip1, ';');
        std::getline(ss, skip2, ';');
        std::getline(ss, skip3, ';');
        std::getline(ss, skip4, ';');
        /// std::getline(ss, skip5, ';');

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
        string skip1, skip2, embedding, left_attr, right_attr;
        getline(ss, skip1, ';');
        getline(ss, skip2, ';');
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

void batch_process_queries(
    clustered_hybrid_search::PredictiveRangeHNSW<float> *alg_query_aware,
    float *query_data,
    std::vector<std::pair<int, int>> &range_queries_meta_data,
    std::unordered_map<unsigned int, int> &meta_data,
    int dim,
    size_t num_threads,
    std::unordered_map<std::string, std::string> &constants)
{
    // Define EF values

    std::vector<int> ef_values = parseIntList(constants.at("EFS"));

    size_t batch_size = std::stoi(constants.at("BATCH_OF_QUERIES"));
    size_t total_elements = alg_query_aware->max_elements_;
    size_t num_batches = (range_queries_meta_data.size() + batch_size - 1) / batch_size;

    std::unordered_map<int, double> totalTimePerEf;

    std::string filter_path_base = constants.at("FILTER_PATH");

    // std::cout << "Total queries: " << range_queries_meta_data.size()
    //           << ", Batch size: " << batch_size
    //           << ", Total batches: " << num_batches << std::endl;

    // ----------- LOOP OVER EF VALUES -----------
    for (int ef : ef_values)
    {

        alg_query_aware->setEf(ef);
        float popularity_threshold = std::stof(constants.at("POPULARITY_THRESHOLD")); // Tuneable threshold for popularity
        alg_query_aware->setPopularityThreshold(popularity_threshold);

        for (size_t b = 0; b < num_batches; b++)
        {

            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, range_queries_meta_data.size());

            std::string filter_file = filter_path_base + std::to_string(start) + ".bin";
            std::vector<char> filter_ids_map;

            if (fileExists(filter_file))
            {
                filter_ids_map =
                    loadFilterMap(filter_file, (end - start) * total_elements);
            }

            else
            {
                // Compute fresh
                std::cout << "💾 Saving " << start << " to cache" << std::endl;
                filter_ids_map.resize((end - start) * total_elements);

                for (size_t i = start; i < end; i++)
                {

                    std::pair<int, int> range_attribute = range_queries_meta_data[i];
                    // std::cout << "Processing query " << i + 1 << "/" << queries
                    ParallelFor(0, alg_query_aware->max_elements_, num_threads, [&](size_t row, size_t threadId)
                                {
                   
                    //bool match_found = (attribute == meta_data_attributes[row]);
                    const auto &row_attributes = meta_data[row];

                        bool match_found = false;

                        
                                    if ((row_attributes >= range_attribute.first && row_attributes <= range_attribute.second))
                                    {
                                        match_found = true;

                                    }

                   //  (attribute >= meta_data_attributes[row]);
                    filter_ids_map[(i - start) * total_elements + row] = match_found; });
                }
                saveFilterMap(filter_ids_map, filter_file);
                std::cout << "💾 Saved filter map for batch " << b + 1 << " to cache" << std::endl;
            }

            // Apply filter
            alg_query_aware->predicateCondition(filter_ids_map.data());

            // ----------- Run queries -----------
            auto batch_start_time = std::chrono::high_resolution_clock::now();

            std::string folder_path = constants.at("RESULTS");
            create_directory_if_not_exists(folder_path);

            ParallelFor(start, end, std::stoi(constants.at("NUM_THREADS")),
                        [&](size_t i, size_t)
                        {
                            alg_query_aware->search(
                                query_data + (i * dim),
                                i, start,
                                range_queries_meta_data[i],
                                std::stoi(constants.at("TOP_K")), folder_path); // topK
                        });

            auto batch_end_time = std::chrono::high_resolution_clock::now();

            double duration_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    batch_end_time - batch_start_time)
                    .count() /
                1000.0;

            totalTimePerEf[ef] += duration_ms;

            std::cout << "Batch " << b + 1
                      << " (ef=" << ef << ") processed in "
                      << duration_ms << " ms" << std::endl;
        }
    }

    // ----------- Final stats per EF -----------
    for (int ef : ef_values)
    {
        double total_queries = static_cast<double>(range_queries_meta_data.size());
        double total_seconds = totalTimePerEf[ef] / 1000.0;
        double qps = total_queries / total_seconds;

        std::cout << "\nEF = " << ef
                  << " => Total Queries: " << total_queries
                  << ", Total Time: " << total_seconds << " s"
                  << ", QPS: " << qps << std::endl;
    }
}

std::unordered_map<std::string, std::string>
reading_constants(const std::string &path)
{
    std::unordered_map<std::string, std::string> constants;
    std::ifstream file(path);

    if (!file)
    {
        std::cerr << "Error opening file!\n";
        return constants;
    }

    std::string line;

    while (std::getline(file, line))
    {
        // 1. Remove leading spaces
        line.erase(0, line.find_first_not_of(" \t"));

        // 2. Skip empty lines
        if (line.empty())
            continue;

        // 3. Skip full-line comments
        if (line[0] == '#')
            continue;

        // 4. Remove inline comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos);
        }

        // 5. Parse key and value
        std::istringstream iss(line);
        std::string key, value;

        if (iss >> key >> value)
        {
            constants[key] = value;
        }
    }

    return constants;
}

std::vector<int> parseIntList(const std::string &str)
{
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, ','))
    {
        result.push_back(std::stoi(item));
    }

    return result;
}
