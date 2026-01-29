#include "../../hnswlib/hnswlib.h"

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
#include <iostream>
#include <random>

using namespace std;
pair<vector<vector<float>>, vector<string>> reading_files(const string &file_path, int &dim, bool queries);



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

int main()
{

    int dim = 128;
    // Dimension of the elements

    int M = 16;
    // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 64;
    // Controls index search speed/build speed tradeoff
    int num_threads = 40; // Number of threads for operations with index

    // For True if you have already index it is only for construction of index
    hnswlib::L2Space space(dim);

   

    auto [embeddings, string_ids] = reading_files("/home/aa5f25/siftsmall/siftsmall_docs.csv", dim, false);
    int max_elements = embeddings.size();

    // Map int -> string IDs
    unordered_map<int, string> id_map;
    float *data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++)
    {
        for (int j = 0; j < dim; j++)
            data[i * dim + j] = embeddings[i][j];
        id_map[i] = string_ids[i];
        
    }
    //  qwery_aware::QweryAwareHNSW<float> *alg_query_aware = new qwery_aware::QweryAwareHNSW<float>(&space, max_elements);

    hnswlib::HierarchicalNSW<float> *alg_query_aware =
        new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    // Build HNSW index
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId)
                { alg_query_aware->addPoint((void *)(data + dim * row), (int)row); });
    alg_query_aware->saveIndex("/home/aa5f25/Index/index.bin");
    delete[] data;
    delete alg_query_aware;
}

// Read embeddings from CSV-like file for Point predicate
pair<vector<vector<float>>, vector<string>> reading_files(const string &file_path, int &dim, bool queries = false)
{
    cout << "Reading file: " << file_path << endl;
    ifstream file(file_path);
    vector<vector<float>> total_embeddings;
    vector<string> attributes; // store string IDs
    string line;
    getline(file, line); // skip header
    while (getline(file, line))
    {

        stringstream ss(line);

        string embedding, attribute, skip;
        if (!queries)
        {

            // getline(ss, skip, ';');
            getline(ss, embedding, ';');
            getline(ss, attribute, ';');
            //  getline(ss, embedding, ';');
        }
        else
        {
            // getline(ss, skip, ';');
            getline(ss, embedding, ';');
            getline(ss, attribute, ';');
        }
        if (!isNullOrEmpty(embedding))
        {
            vector<float> embeddingVector = splitToFloat(embedding, ',');
            if (embeddingVector.size() == dim)
            {
                total_embeddings.push_back(embeddingVector);
                attributes.push_back(attribute); // store string ID
            }
        }
    }
    return {total_embeddings, attributes};
}
