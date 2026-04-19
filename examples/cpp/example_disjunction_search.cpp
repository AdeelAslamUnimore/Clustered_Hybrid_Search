#include "../../hnswlib/hnswlib.h"
#include "../../predictive_hybrid_queries/predictive_disjunction_queries_clustering.h"
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

using namespace std;
std::unordered_map<unsigned int, std::pair<std::vector<std::string>, int>> reading_meta_data(const string &file_path);
pair<vector<vector<float>>, std::vector<std::pair<std::vector<std::string>, std::pair<int, int>>>> reading_queries(const string &file_path, int &dim);
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
// Helper fucntion to trim whitespace
string trim(const string &s)
{
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == string::npos) ? "" : s.substr(start, end - start + 1);
}
vector<string> split_and_clean(const string &s)
{

    // Split attribute by ',' into vector
    std::vector<std::string> attr_vec;
    std::stringstream attr_ss(s);
    std::string token;
    while (std::getline(attr_ss, token, ','))
    {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
        token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);

        if (!token.empty())
            attr_vec.push_back(token);
    }
    return attr_vec;
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
void batch_process_queries(
    clustered_hybrid_search::PredictiveDisjunctionHNSW<float> *alg_query_aware,
    float *query_data,
    std::vector<std::pair<std::vector<std::string>, std::pair<int, int>>> &queries_meta_data, // Updated this line
    std::unordered_map<unsigned int, std::pair<std::vector<std::string>, int>> &meta_data,
    int dim,
    size_t num_threads);
int main(int argc, char const *argv[])
{
    /* code */
    int dim = 1024;

    std::unordered_map<unsigned int, std::pair<std::vector<std::string>, int>> meta_data = reading_meta_data("/scratch/aa5f25/datasets/yt8m/yt_data.csv");

    int max_elements = meta_data.size();

    hnswlib::L2Space space(dim);

    clustered_hybrid_search::PredictiveDisjunctionHNSW<float> *alg_query_aware = new clustered_hybrid_search::PredictiveDisjunctionHNSW<float>(&space, max_elements, "/scratch/aa5f25/datasets/yt8m/index.bin", meta_data); // Load existing index

    // alg_query_aware->clustering_for_cms_and_cdf_filtering(100);
    auto start = std::chrono::high_resolution_clock::now();

    alg_query_aware->clustering_for_cms_and_cdf_filtering_for_quantile_regression(10000);

    //alg_query_aware->clustering_for_cms_and_cdf_filtering_for_pieceWise_regression(20000);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Start time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count()
              << " ms\n";

    std::cout << "End time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch()).count()
              << " ms\n";

    std::cout << "Execution time: " << duration.count() << " seconds\n";
    exit(0);

    pair<vector<vector<float>>, std::vector<std::pair<std::vector<std::string>, std::pair<int, int>>>> query_reading_results = reading_queries("/scratch/aa5f25/datasets/yt8m/Queries_for_conjunctive_disjunctive.csv", dim);

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

    batch_process_queries(
        alg_query_aware,
        query_data,
        query_reading_results.second,
        meta_data,
        dim, 40);
    // /*num_threads=*/8);
    // alg_query_aware->freeMemory();

    delete[] query_data;
    delete alg_query_aware;
}
// Reading the Meta data
std::unordered_map<unsigned int, std::pair<std::vector<std::string>, int>>
reading_meta_data(const std::string &file_path)
{
    std::cout << "Reading meta_data from " << file_path << std::endl;

    std::ifstream file(file_path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + file_path);

    std::unordered_map<unsigned int, std::pair<std::vector<std::string>, int>> meta_data;
    std::string line;

    // Skip header
    std::getline(file, line);

    size_t line_count = 0;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string embedding, attribute_int, attribute_str;
        std::string skip1, skip2, skip3, skip4;

        // CSV format: embedding;attribute;range_attribute
        std::getline(ss, skip1, ';');
        std::getline(ss, embedding, ';');
        std::getline(ss, skip2, ';');
        std::getline(ss, attribute_str, ';');
        std::getline(ss, skip3, ';');
        std::getline(ss, attribute_int, ';');
        std::getline(ss, skip4, ';');

        // Convert range_attribute to int
        int range_attribute = 0;
        try
        {
            range_attribute = std::stoi(attribute_int); // Change it to range_attribute_Str
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error converting range_attribute to int at line " << line_count << ": " << e.what() << std::endl;
            range_attribute = 0; // fallback
        }

        // Split attribute by ',' into vector
        std::vector<std::string> attr_vec;
        std::stringstream attr_ss(attribute_str);
        std::string token;
        while (std::getline(attr_ss, token, ','))
        {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
            token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);

            if (!token.empty())
                attr_vec.push_back(token);
        }

        meta_data[line_count] = {attr_vec, range_attribute};

        line_count++;
    }

    return meta_data;
}

// Read embeddings and combined point/range predicates
pair<vector<vector<float>>, std::vector<std::pair<std::vector<std::string>, std::pair<int, int>>>>
reading_queries(const string &file_path, int &dim)
{
    cout << "Reading file: " << file_path << endl;
    ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open query file: " + file_path);
    }

    vector<vector<float>> total_embeddings;
    // This now matches your return type signature exactly
    std::vector<std::pair<std::vector<std::string>, std::pair<int, int>>> all_query_predicates;

    string line;
    getline(file, line); // skip header

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream ss(line);
        string embedding_str, attribute_str, range_str1, range_str2;
        string skip1, skip2;

        // CSV format: embedding;point_attributes;range_start;range_end
        getline(ss, skip1, ';'); // Skip ID or other non-essential field
        getline(ss, embedding_str, ';');
        getline(ss, skip2, ';'); // Skip audio

        getline(ss, range_str1, ';');
        getline(ss, range_str2, ';');
        getline(ss, attribute_str, ';');

        if (!isNullOrEmpty(embedding_str))
        {
            vector<float> embeddingVector = splitToFloat(embedding_str, ',');

            if (embeddingVector.size() == static_cast<size_t>(dim))
            {
                total_embeddings.push_back(embeddingVector);

                // 1. Process point attributes (strings)
                vector<string> attributeTokens = split_and_clean(attribute_str);

                // 2. Process range attributes (integers)
                int range_attr1 = 0, range_attr2 = 0;
                try
                {
                    if (!range_str1.empty())
                        range_attr1 = std::stoi(range_str1);
                    if (!range_str2.empty())
                        range_attr2 = std::stoi(range_str2);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Warning: Integer conversion failed for line: " << line << " | " << e.what() << std::endl;
                }

                // 3. Store the combined metadata pair
                all_query_predicates.push_back({attributeTokens, {range_attr1, range_attr2}});
            }
        }
    }

    return {total_embeddings, all_query_predicates};
}

void batch_process_queries(clustered_hybrid_search::PredictiveDisjunctionHNSW<float> *alg_query_aware, float *query_data, std::vector<std::pair<std::vector<std::string>, std::pair<int, int>>> &queries_meta_data, std::unordered_map<unsigned int, std::pair<std::vector<std::string>, int>> &meta_data, int dim, size_t num_threads)

{
    std::vector<int> ef_values = {20, 40, 80, 100, 200, 300, 500, 800, 1000, 1200, 1500, 1700, 1900, 2100, 2300}; //{10, 20, 60, 200, 400, 800, 1000, 1200, 1300};

    // Track total time per ef
    std::unordered_map<int, double> totalTimePerEfs;

    for (int ef : ef_values)
    {
        size_t batch_size = 1000;
        size_t total_elements = alg_query_aware->max_elements_;
        size_t num_batches = (queries_meta_data.size() + batch_size - 1) / batch_size;

        // Set ef once per run
        alg_query_aware->setEf(ef);

        for (size_t b = 0; b < num_batches; b++)
        {
            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, queries_meta_data.size());

            std::string filter_file =
                "/scratch/aa5f25/datasets/yt8m/filters/filter_batch_" +
                std::to_string(start) + ".bin";
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
                    // const std::string &attribute = queries_meta_data[i];
                    vector<string> query_attributes = queries_meta_data[i].first;
                    std::pair<int, int> range_attribute = queries_meta_data[i].second;
                    // std::cout << "Processing query " << i + 1 << "/" << queries
                    ParallelFor(0, alg_query_aware->max_elements_, num_threads, [&](size_t row, size_t threadId)
                                {
                   
                    //bool match_found = (attribute == meta_data_attributes[row]);
                    const auto &row_attributes = meta_data[row];

                        bool match_found = false;

                        // Outer loop: each attribute in the query
                            for (const auto &q_attr : query_attributes)
                            {
                                // Inner loop: each attribute in the current metadata row
                                for (const auto &row_attr : row_attributes.first)
                                {
                                    if ((q_attr == row_attr)||(row_attributes.second >= range_attribute.first && row_attributes.second <= range_attribute.second))
                                    {
                                        match_found = true;
                                        break; // Break inner loop
                                    }
                                }
                                if (match_found) break; // Break outer loop if a match was found
                            }

                   //  (attribute >= meta_data_attributes[row]);
                    filter_ids_map[(i - start) * total_elements + row] = match_found; });
                }
                saveFilterMap(filter_ids_map, filter_file);
                std::cout << "💾 Saved filter map for batch " << b + 1 << " to cache" << std::endl;
            }

            // Apply filter
            alg_query_aware->predicateCondition(filter_ids_map.data());

            // ------------------ Run queries for this batch ------------------
            auto batch_start_time =
                std::chrono::high_resolution_clock::now();

            // Determine threads
            size_t num_threads = std::thread::hardware_concurrency();
            if (const char *cpus = std::getenv("SLURM_CPUS_PER_TASK"))
            {
                num_threads = std::stoi(cpus);
            }

            // std::cout << "Running batch " << b + 1 << "/" << num_batches
            //           << " with ef=" << ef
            //           << " using " << num_threads << " threads" << std::endl;

            ParallelFor(
                start,
                end,
                40,
                [&](size_t row, size_t)
                {
                    alg_query_aware->search((query_data + row * dim), row, start, queries_meta_data[row].second, queries_meta_data[row].first, 10);

                    // alg_query_aware->search((query_data + row * dim),
                    //                         row,
                    //                         start,
                    //                         queries_meta_data[row],
                    //                         10);

                    // alg_query_aware->groundTruthBatch(query_data + row * dim,
                    //                                   10,
                    //                                   row,
                    //                                   total_elements,
                    //                                   start);
                });

            auto batch_end_time =
                std::chrono::high_resolution_clock::now();

            auto duration_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    batch_end_time - batch_start_time)

                    .count() /
                1000.0;

            totalTimePerEfs[ef] += duration_ms;
            // alg_query_aware->result_print();
        }
    }

    // ------------------ Final statistics summary ------------------

    for (int ef : ef_values)
    {
        double total_queries = static_cast<double>(queries_meta_data.size());
        double total_seconds = totalTimePerEfs[ef] / 1000.0;
        double qps = total_queries / total_seconds;

        std::cout << "Search Time for efSearch=" << ef
                  << " => Total Queries: " << total_queries
                  << ", Total Time: " << total_seconds << "s"
                  << ", QPS: " << qps << std::endl;
    }
}