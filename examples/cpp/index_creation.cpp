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
 * @brief Reads data from the specified file and extracts embeddings and predicates.
 *
 * This function reads a file containing data in a specific format, where each line consists of an embedding vector (represented as a comma-separated string)
 * and optionally additional information (e.g., predicates like clinical areas). It processes each line to convert the embedding into a vector of floats and
 * stores it in a vector if the embedding has the correct dimensionality (specified by the 'dim' parameter). Optionally, additional predicates can also be
 * processed and stored. The function returns a pair containing two vectors:
 *  - A vector of vectors representing the embeddings (total_embeddings).
 *  - A vector of vectors of char pointers representing predicates (predicates).
 *
 * The file format is assumed to be CSV-like, where each line is divided by semicolons, and the first column contains the embedding data. The embeddings
 * are expected to be comma-separated within each line.
 *
 * @param file_path The path to the file to be read.
 * @param dim A reference to the dimensionality of the embeddings. This is used to ensure the embeddings have the correct size (e.g., 768 for clinical data).
 *
 * @return A pair of vectors: one for embeddings (total_embeddings) and one for predicates (predicates).
 */
pair<vector<std::vector<float>>, vector<vector<char *>>> reading_files(std::string file_path, int &dim)
{
    //   "/data4/hnsw/documents.csv"
    std::ifstream file(file_path);
    std::vector<std::vector<float>> total_embeddings;
    std::vector<std::vector<char *>> predicates;
    std::string line;
    // Read the file line by line
    int count = 0;
    getline(file, line);
    int test_counter = 0;

    while (getline(file, line))
    {

        std::stringstream ss(line);
        std::string embedding;

        getline(ss, embedding, ';');

        // Initialize vectors

        vector<float> embeddingVector;
        vector<char *> clinicalAreasVector;

        if (!isNullOrEmpty(embedding)) // && !isNullOrEmpty(dateCreated)
        {
            // Split clinicalAreas and embedding
            embeddingVector = splitToFloat(embedding, ',');
            // clinicalAreasVector = splitString(dateCreated, ',');
            // Here do one thing

            if (embeddingVector.size() == dim) // 768 for clinical data
            {

                total_embeddings.push_back(embeddingVector);
                // predicates.push_back(clinicalAreasVector);
            }
        }
    }

    file.close();
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<char *>>> data_vectors(total_embeddings, predicates);
    return data_vectors;
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
    int dim = std::stoi(constants["DIM"]); // Dimension of the elements
    int M = std::stoi(constants["M"]);     // Tightly connected with internal dimensionality of the data
                                           // strongly affects the memory consumption
    int ef_construction = std::stoi(constants["EFC"]);

    pair<vector<std::vector<float>>, vector<vector<char *>>> data_vectors = reading_files(constants["DATASETFILE"], dim);
    int max_elements = data_vectors.first.size();
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    float *data = new float[dim * max_elements];

    int index_of_vector = 0;
    ////  vector<std::vector<float>> index_items = data_vectors.first;
    for (const auto &vec : data_vectors.first)
    {
        // Iterate over each float in the current vector
        for (float value : vec)
        {
            if (index_of_vector < dim * max_elements)
            { // Check to avoid out-of-bounds access
                data[index_of_vector] = value;
                index_of_vector++;
            }
            else
            {
                std::cerr << "Error: data array out of bounds" << std::endl;
                break;
            }
        }
    }
    ParallelFor(0, max_elements, 40, [&](size_t row, size_t threadId)
                {
                    alg_hnsw->addPoint((void *)(data + dim * row), row);
                });

    alg_hnsw->saveIndex(constants["INDEXPATH"]);
}
