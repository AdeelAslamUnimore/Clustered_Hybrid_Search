#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/count_min_sketch_min_hash.hpp"
#include "../../hnswlib/BPlusTree.h"

#include "../../hnswlib/hashutil.h"
#include "../../hnswlib/Node.h"

#include <stdint.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <map>
#include <iostream>
#include <immintrin.h>
#include <iomanip> // for std::put_time
#include <sstream> // for std::istringstream
// C++17 or later
#include <cstring>
#include <thread>
#include <variant>
using namespace std;

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
// Define a variant to hold different types
std::unordered_map<std::string, std::string> reading_constants()
{
    std::unordered_map<std::string, std::string> constants; // Store values as strings
    std::ifstream file("../examples/cpp/constants_and_filepaths.txt");

    if (!file)
    {
        std::cerr << "Error opening file!" << std::endl;
        return constants;
    }

    std::string key, value;

    while (file >> key >> value)
    {
        constants[key] = value;
    }

    file.close();

    return constants;
}

// Function to split a string by a delimiter
vector<char *> splitString(const std::string &str, char delimiter)
{
    vector<char *> tokens;
    string token;
    istringstream tokenStream(str);

    while (getline(tokenStream, token, delimiter))
    {
        char *cstr = new char[token.length() + 1];
        strncpy(cstr, token.c_str(), token.length() + 1);
        tokens.push_back(cstr);
    }

    return tokens;
}

// Function to split a string by a delimiter and convert to doubles
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
// Function to check the nulls
bool isNullOrEmpty(const string &str)
{
    return str.empty() || str == "null";
}

/**
 * @brief Reads metadata from a given file and stores it in a hash table.
 *
 * This function reads a file containing metadata where each row represents a unique record.
 * The row index acts as an identifier and is used as the key in an unordered_map (hash table).
 * The corresponding value is a vector of C-style strings (char*), storing metadata fields.
 *
 * @param file_path Path to the metadata file.
 * @return std::unordered_map<unsigned int, std::vector<char*>>
 *         A hash table where:
 *         - Key: Row index (identifier for each record)
 *         - Value: Vector of metadata fields (stored as char*)
 */
std::unordered_map<unsigned int, std::vector<char *>> reading_metaData(std::string file_path)
{

    std::unordered_map<unsigned int, std::vector<char *>> hashTable;

    std::ifstream file(file_path);

    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return hashTable;
    }

    std::string line;
    int rowIndex = 0;

    // Read the file line by line
    int recordNum = 0;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        if (!line.empty() && line.front() == '"' && line.back() == '"')
        {
            line = line.substr(1, line.size() - 2);
        }

        std::stringstream ss(line);
        std::string meta_data;

        getline(ss, meta_data, ';');

        // Vector of char* to store each split string
        std::vector<char *> meta_data_array;
        std::stringstream stream(meta_data);

        std::string part;
        // Split by comma and add each part to the vector
        while (std::getline(stream, part, ','))
        {
            // Trim whitespace from the part
            part.erase(0, part.find_first_not_of(" \t\n\r"));
            part.erase(part.find_last_not_of(" \t\n\r") + 1);

            char *cstr = strdup(part.c_str()); // Convert std::string to char*
            meta_data_array.emplace_back(cstr);
        }

        hashTable[rowIndex] = meta_data_array;

        rowIndex++;
    }

    file.close();
    return hashTable;
}

int main()
{

    std::unordered_map<std::string, std::string> constants = reading_constants();
    
    int dim = std::stoi(constants["DIM"]);
   
    int cluster_size = std::stoi(constants["CLUSTER_SIZE"]);
   
    double popularity_threshold = std::stod(constants["POPULARITY_THRESHOLD_POINT"]);
    hnswlib::L2Space space(dim);
    std::unordered_map<unsigned int, std::vector<char *>> metaData = reading_metaData(constants["META_DATA_PATH"]);
    int max_elements = metaData.size();
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, constants["INDEX_PATH"], metaData, max_elements);
    // Construct the clusters and maintain the count-min sketch
    alg_hnsw->clustering_for_cdf_range_filtering_int(cluster_size);

    // Reading the queries with attributes
    std::ifstream file(constants["QUERIES_PATH"]);
    std::vector<unsigned int> left_range;
    std::vector<unsigned int> right_range;
    std::vector<std::vector<float>> total_embeddings;
    std::vector<std::vector<char *>> query_preds;
    // std::vector<std::vector<std::string>> query_preds_string;
    std::string line;
    // Read the file line by line
    int count = 0;
    getline(file, line);
    int test_counter = 0;

    while (getline(file, line))
    {

        std::stringstream ss(line);
        std::string embedding, right_range, left_range;

        getline(ss, embedding, ';');

        getline(ss, left_range, ';');
        getline(ss, right_range, ';');

        unsigned int left_r = std::stoul(left_range);
        unsigned int right_r = std::stoul(right_range);

        vector<float> embeddingVector;

        if (!isNullOrEmpty(embedding))
        {
            // Split clinicalAreas and embedding
            embeddingVector = splitToFloat(embedding, ',');
            // Here do one thing

            if (embeddingVector.size() == dim) // 768 for clinical data
            {

                left_range.push_back(left_r);
                right_range.push_back(right_r);
                total_embeddings.emplace_back(embeddingVector);
            }
        }
    }

    file.close();

    // Total EFS which further used for exploratory search on HNSW
    int total_efs[] = {
        20,
        40,
        60,
        100,
        200,
        300,
        400,
        500,
        600,
        800,
        1000, 1200, 1500, 1700, 1900, 2100

    };
    // Number of queries
    int size_of_query_items = total_embeddings.size();

    float *query_data = new float[dim * size_of_query_items];
    int index_of_query_vector = 0;

    for (const auto &vec : total_embeddings)
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

    for (int i = 0; i < sizeof(total_efs) / sizeof(total_efs[0]); i++)
    {

        alg_hnsw->setEf(total_efs[i]);

        auto start = std::chrono::high_resolution_clock::now();
        // 40 is the number of threads and 10 is the top 10
        ParallelFor(0, size_of_query_items, 40, [&](size_t row, size_t threadId)
                    { alg_hnsw->rangeSearch(query_data + (row * dim), 10, left_range[row], right_range[row], row, total_efs[i], popularity_threshold, constants["RESULTFOLDER"]); });

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double qps = static_cast<double>(duration) / 1000.0;
        double total_queries = static_cast<double>(size_of_query_items);
        double res = total_queries / qps;
        // // // Output the time taken
        std::cout << "Time taken for post_filtering in microseconds: " << "For:  " << total_efs[i] << "Time:   " << res << std::endl;

        // This block runs only once to compute ground truth data for evaluation purposes.
        // The `break` ensures it executes only during the first iteration of an outer loop (not shown here).
        // Other parts of the code remain commented, but they do not cause any issues if left as is.
        // for (int j = 0; j < size_of_query_items; j++)
        // {
        //    alg_hnsw->ground_truth_computer_for_predicate(query_data + (i * dim), 15, left_range[i], right_range[i], i, constants["GROUND_TRUTH_FILE"]);
        // }
        // break; // Ensures this block runs only once
    }

    

        delete[] query_data;
    alg_hnsw->freeMemory();
}
