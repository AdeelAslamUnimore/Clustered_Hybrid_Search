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
    std::ifstream file("../exampleFolder/constants.txt");

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
    // Retrieve values
    // std::string DIM = constants["DIM"];
    // std::string CLUSTERSIZE = constants["CLUSTERSIZE"];
    // std::string POPULARITY_THRESHOLD_POINT = constants["POPULARITYTHRESHOLDPOINT"];
    // std::string POPULARITY_THRESHOLD_CDF = constants["POPULARITYTHRESHOLDCDF"];
    // std::string INDEX_PATH = constants["INDEXPATH"];
    // std::string META_DATA_PATH = constants["METADATAPATH"];
    // std::string QUERIES_PATH=constants["QUERIESPATH"];
    int dim = std::stoi(constants["DIM"]);
    int cluster_size = std::stoi(constants["CLUSTERSIZE"]);
    int popularity_threshold = std::stoi(constants["POPULARITYTHRESHOLDPOINT"]);
    hnswlib::L2Space space(dim);
    std::unordered_map<unsigned int, std::vector<char *>> metaData = reading_metaData(constants["METADATAPATH"]);
    int max_elements = metaData.size();
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, constants["INDEXPATH"], metaData, max_elements);
    // Construct the clusters and maintain the count-min sketch
    alg_hnsw->clustering_and_maintaining_sketch(cluster_size);
    // Reading the queries with attributes
    std::ifstream file(constants["QUERIESPATH"]);
    std::vector<std::vector<float>> total_embeddings;
    std::vector<std::vector<char *>> query_attributes;
    std::string line;
    // Read the file line by line
    // Header reader; which need to be split
    getline(file, line);
    int test_counter = 0;
    while (getline(file, line))
    {
        std::stringstream ss(line);
        std::string embedding, attribute;
        getline(ss, attribute, ';');
        getline(ss, embedding, ';');
        // Vector of char* to store each split string
        std::vector<char *> query_attributes_;
        std::stringstream areaStream(attribute);
        std::string part;
        // Split by comma and add each part to the vector
        while (std::getline(areaStream, part, ','))
        {
            char *cstr = strdup(part.c_str()); // Convert std::string to char*
            query_attributes_.push_back(cstr);
        }

        vector<float> embeddingVector;
        if (!isNullOrEmpty(embedding))
        {
            // Split attributes and embedding
            embeddingVector = splitToFloat(embedding, ',');
            // Here do one thing

            if (embeddingVector.size() == dim) // 768 for clinical data
            {

                total_embeddings.emplace_back(embeddingVector);
                query_attributes.push_back(query_attributes_);
            }
        }
    }

    // Close the file
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
        ParallelFor(0, size_of_query_items, 40, [&](size_t row, size_t threadId)
                    { alg_hnsw->clustered_based_exhaustive_search(query_data + (row * dim), 10, query_attributes[row], row, total_efs[i], popularity_threshold, constants["RESULTFOLDER"]); });

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
        for (int j = 0; j < size_of_query_items; j++)
        {
            alg_hnsw->ground_truth_point_predicate(query_data + (j * dim), 15, query_attributes[j], j, constants["GROUNDTRUTHFILE"] );
        }
        break; // Ensures this block runs only once
    }
    // un comment only when you need to compute the ground truth as well Run it once

    alg_hnsw->freeMemory();

    // Iterate through each vector in query_attributes
    for (auto &inner_vector : query_attributes)
    {
        // Iterate through each char* in the inner vector and free the allocated memory
        for (auto &str : inner_vector)
        {
            free(str); // Free the char* memory
        }
    }
}
