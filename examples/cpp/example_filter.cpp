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

using namespace std;
void testingG(int &a, unsigned int &b);

std::string generateRandomString(int len);
vector<char *> splitString(const string &str, char delimiter);
vector<float> splitToFloat(const string &str, char delimiter);
int ranking(int number, int pos);
bool isNullOrEmpty(const string &str);
void bench_mark();
float l2_distance(const std::vector<float> &a, const std::vector<float> &b);
pair<vector<std::vector<float>>, vector<vector<char *>>> reading_files(std::string file_path, bool flag_read);
std::unordered_map<unsigned int, std::vector<char *>> reading_metaData(std::string file_path);
bool loadCMSFiles(const std::string &folderPath, std::unordered_map<unsigned int, CountMinSketchMinHash> &mapForCMS);

std::map<std::string, std::vector<unsigned int>> reading_range_data(std::string file_path);
// Multi diemensional data
std::unordered_map<unsigned int, std::pair<std::string, std::vector<char *>>> reading_metaData_hybrid_query(std::string file_path);

// void saveToFile(const std::unordered_map<std::string, RangeSearch<std::string>> &searchMap, const std::string &filename);

// std::unordered_map<std::string, RangeSearch<std::string>> loadFromFile(const std::string &filename);

std::map<unsigned int, Vertex<std::string> *> leafNodes();
std::unordered_map<unsigned int, unsigned int> reading_metaData_int(std::string file_path);

char *mem_for_ids_clusters{nullptr};

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
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
// Intrinsic calls

int main()
{

   

    std::map<unsigned int, Vertex<std::string> *> searchMap;

    int dim = 128; // 1024; // Dimension of the elements

    int M = 256;               // Tightly connected with internal dimensionality of the data
                               // strongly affects the memory consumption
    int ef_construction = 600; // Controls index search speed/build speed tradeoff 200
                               // Loading Data from file

    int max_elements = 0;
    std::unordered_map<unsigned int, CountMinSketchMinHash> mapCMS;
    //     // loadCMSFiles("/data4/hnsw/paper/Clusters/CMS_16_32/", mapCMS);
    std::unordered_map<unsigned int, std::pair<std::string, std::vector<char *>>> multi_diemesional_meta_data; //= reading_metaData_hybrid_query("/data4/hnsw/TripClick/mutli_diemensional_meta_data.csv");

    // std::unordered_map<unsigned int, std::vector<char *>> metaData;
    //= reading_metaData("/data4/hnsw/yt8m/Meta_data_views.csv");

    std::unordered_map<unsigned int, unsigned int> metaData_int=reading_metaData_int("/data4/hnsw/yt8m/meta_data_likes.csv");

    //     // // //     // Loading the Index Already computed using structures

    hnswlib::L2Space space(dim);
    bool cluster_read_write = false;
    std::unordered_map<unsigned int, std::vector<char *>> metaData = reading_metaData("/data4/hnsw/TripClick/Meta_data_clinical_Area.csv");

    max_elements = metaData_int.size();
    cout << "Maxim  " << max_elements << endl;

    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, "/data4/hnsw/yt8m/IndexStructure/M_200_efc_600_128.bin", metaData, multi_diemesional_meta_data, metaData_int, max_elements, mapCMS, cluster_read_write, searchMap);
    // exit(0);

    // alg_hnsw->clustering_for_corelation(1000);
      //  alg_hnsw->clustering_and_maintaining_sketch(100000);
    // alg_hnsw->clustering_and_maintaining_sketch_with_disk_optimization(100000);
    // cout<<"Clustering"<<endl;
    // alg_hnsw->clustering_and_maintaining_sketch_test_memory(10000);

    // alg_hnsw->clustering_multidiemensional_range(100000);
    //  alg_hnsw->clustering_for_cdf_range_filtering(100000);
    alg_hnsw->clustering_for_cdf_range_filtering_int(100000);

    std::ifstream file("/data4/hnsw/yt8m/queries_range_likes.csv");
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
        std::string qtext, embedding, right_str, left_str, clinicalArea;
        // unsigned int left_r, right_r; // left_r, right_r, clinicalArea,
        // Assuming columns are separated by semicolons
        // if (flag_read == true)
        getline(ss, qtext, ';');
        getline(ss, clinicalArea, ';');
        getline(ss, embedding, ';');
        // getline(ss, right_str, ';');

        // getline(ss, embedding, ';');

        getline(ss, left_str, ';');
         getline(ss, right_str, ';');
        // getline(ss, embedding, ';');
        // getline(ss, right_r, ';');
        // getline(ss, clinicalArea, ';');
        // Initialize vectors
        unsigned int left_r = std::stoul(left_str);
        unsigned int right_r = std::stoul(right_str);

        // Vector of char* to store each split string
        std::vector<char *> clinicalAreaParts;
        // std:: vector<std::string> clAreas;
        std::stringstream areaStream(clinicalArea);
        std::string part;
        // Split by comma and add each part to the vector

        while (std::getline(areaStream, part, ','))
        {

            char *cstr = strdup(part.c_str()); // Convert std::string to char*
            clinicalAreaParts.push_back(cstr);
        }

        vector<float> embeddingVector;

        if (!isNullOrEmpty(embedding))
        {
            // Split clinicalAreas and embedding
            embeddingVector = splitToFloat(embedding, ',');
            // Here do one thing

            if (embeddingVector.size() == 128) // 768 for clinical data
            {

                left_range.push_back(left_r);
                right_range.push_back(right_r);
                total_embeddings.emplace_back(embeddingVector);

                // query_preds.emplace_back(clinicalAreaParts);
                // query_preds.push_back(clinicalAreaParts);

             //   cout<<"Left::  "<<left_r<<"right:: "<<right_r<<"Embedding:: "<<embeddingVector.size()<<endl;
            }
        }
    }
   // cout << "Q" << query_preds.size() << endl;
    // Close the file
    file.close();

    //  alg_hnsw-> clustering_for_bPlustree_range_filtering(50000, &space);

    //     //  alg_hnsw->test();
    //     //  exit(0);
    //     //     // pair<vector<std::vector<float>>, vector<vector<char *>>> query_vectors = reading_files("/data4/hnsw/paper/paper_queries.csv", true);

    //     //    // vector<std::vector<float>> query_items = query_vectors.first;
    // 20, 40, 60, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500,1700, 1900, 2100, 2400, 2700, 3000, 3200, 3400, 3600,3800, 4000
 
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
        //  cout<<"I am here"<<total_efs[i]<< endl;

        auto start = std::chrono::high_resolution_clock::now();
        ParallelFor(0, size_of_query_items, 40, [&](size_t row, size_t threadId)
                    {
                     //  alg_hnsw-> postFilteringApproachesRange(query_data + (row * dim),10, left_range[row], right_range[row], row, total_efs[i]);
                       // alg_hnsw-> rangeSearch(query_data + (row * dim),10, left_range[row], right_range[row], row, total_efs[i]);

                        //     //             // alg_hnsw->rangeSearch(query_data + (row * dim), 30, left_range[row], right_range[row], row, total_efs[i]);
                        //     //             //     / alg_hnsw->addPoint((void *)(data + dim * row), row);
                       // alg_hnsw->clustered_based_exhaustive_search(query_data + (row * dim), 10, query_preds[row], row, total_efs[i]);
                        //     //             //     alg_hnsw->addPointWithMetaData(data + (row * dim), row, data_vectors.second[row]);
                        // //    alg_hnsw->search_hybrid_range(query_data + (row * dim), 48, left_range[row], right_range[row],query_preds[row], row, total_efs[i]);
                    });
        // alg_hnsw-> counter_disk_access();

        // for (int i = 0; i < size_of_query_items; i++)
        // {
        //     // cout<<query_preds[i]<<"   "<< endl;
        //     // alg_hnsw->ground_truth_computer_for_multiattribute(query_data + (i * dim), 10, left_range[i], right_range[i], query_preds[i],i);
        //    // alg_hnsw->ground_truth_point_predicate(query_data + (i * dim), 15, query_preds[i], i);
        //      alg_hnsw->ground_truth_computer_for_predicate(query_data + (i * dim), 15, left_range[i], right_range[i], i);
        //     //  alg_hnsw->computing_data_for_finding_corelation(query_data + (i * dim), 15, query_preds[i], i);
          
        // }
        // break;
        // //         alg_hnsw->rangeSearch(query_data + (i * dim),10, left_range[i], right_range[i], i);
        //         break;
        // //          alg_hnsw->ground_truth_computer_for_predicate(query_data + (i * dim), 15, left_range[i], right_range[i], i);
        // alg_hnsw-> ground_truth_point_predicate(query_data + (i * dim), 15, query_preds[i], i);
        // //      alg_hnsw->clustered_based_exhaustive_search(query_data + (i * dim), 10, query_preds[i], i);

        //     }
        //     //alg_hnsw-> postFilteringApproachesRange(query_data + (i * dim),39, left_range[i], right_range[i], i);
        //   //break;
        //     // alg_hnsw->ground_truth_computer_for_predicate(query_data + (i * dim), 15, left_range[i], right_range[i], i);

        //  // alg_hnsw->clustered_based_exhaustive_search(query_data + (i * dim), 39, query_preds[i], i);
        // }
        // alg_hnsw->search_hybrid_range(query_data + (i * dim), 39, left_range[i], right_range[i],query_preds[i], i);
        // cout<<i<<endl;
        // alg_hnsw->ground_truth_computer_for_multiattribute(query_data + (i * dim), 10, left_range[i], right_range[i], query_preds[i],i);
        //  exit(0);
        // cout<< left_range[i]<<"Right....."<<right_range[i]<<endl;
        // alg_hnsw->postFilteringApproachesRange(query_data + (i * dim), 39, left_range[i], right_range[i], i);

        // alg_hnsw-> postFiltering_range_bPlus_tree(query_data + (i * dim),39, left_range[i], right_range[i], i);

        //  alg_hnsw->rangeSearch(query_data + (i * dim),10, left_range[i], right_range[i], i);
        // alg_hnsw-> ground_truth_point_predicate(query_data + (i * dim), 15, query_preds[i], i);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double qps = static_cast<double>(duration) / 1000.0;
        double res = 1500.0 / qps;
        // // // Output the time taken
        std::cout << "Time taken for post_filtering in microseconds: " << "For:  " << total_efs[i] << "Time:   " << res << std::endl;
        // break;
    }

    alg_hnsw->freeMemory();
    //  delete alg_hnsw;

    // Iterate through all elements in node_map
    for (const auto &pair : searchMap)
    {
        // pair.first is the unsigned int key
        // pair.second is the Vertex<string>* pointer
        delete pair.second; // Delete the Vertex object
    }

    return 0;
}

pair<vector<std::vector<float>>, vector<vector<char *>>> reading_files(std::string file_path, bool flag_read)
{

    //   "/data4/hnsw/documents.csv"
    std::ifstream file(file_path);
    std::vector<std::vector<float>> total_embeddings;
    std::vector<std::vector<char *>> predicates;
    std::string line;
    // Read the file line by line
    int count = 0;
    // getline(file, line);
    int test_counter = 0;

    while (getline(file, line))
    {

        std::stringstream ss(line);
        std::string qtext, dateCreated, clinicalAreas, embedding;

        // Assuming columns are separated by semicolons
        // if (flag_read == true)
        //     getline(ss, qtext, ';');

        getline(ss, dateCreated, ';');
        getline(ss, clinicalAreas, ';');
        getline(ss, embedding, ';');
        //   getline(ss, clinicalAreas, ';');
        // Initialize vectors

        vector<float> embeddingVector;
        vector<char *> clinicalAreasVector;

        if (!isNullOrEmpty(embedding) && !isNullOrEmpty(dateCreated))
        {
            // Split clinicalAreas and embedding
            embeddingVector = splitToFloat(embedding, ',');
            clinicalAreasVector = splitString(dateCreated, ',');
            // Here do one thing

            if (embeddingVector.size() == 768) // 768 for clinical data
            {
                total_embeddings.push_back(embeddingVector);
                predicates.push_back(clinicalAreasVector);
            }
        }
        test_counter++;
        if (test_counter % 100000 == 0)
        {
            cout << "Counter" << test_counter << endl;
        }
    }

    // Close the file
    file.close();
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<char *>>> data_vectors(total_embeddings, predicates);
    return data_vectors;
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
// Also make sure to remove the memory address occupied by char* after insertion.

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

bool isNullOrEmpty(const string &str)
{
    return str.empty() || str == "null";
}

float l2_distance(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("Vectors must be of the same dimension");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

int ranking(int number, int pos)
{
    // Create a mask to zero out bits after the given position
    int mask = (1 << (pos + 1)) - 1; // Creates mask like 000001111 for pos=4
    int maskedNumber = number & mask;

    // Use __builtin_popcount to count the number of set bits
    return __builtin_popcount(maskedNumber);
    // return 0;
}

void testingG(int &id, unsigned int &cluster_id_)
{

    size_t byte_offset = id * sizeof(int);
    unsigned int *address_of_index = (unsigned int *)(mem_for_ids_clusters + byte_offset);

    // Dereference the pointer to get the integer value use unsigned int other wise signed may cause issue for most significant bits
    unsigned int value_at_index = *address_of_index;

    // Check the first bit (least significant bit)
    bool is_first_bit_set = (value_at_index & 1) != 0;

    if (is_first_bit_set)
    {
        unsigned int clusterID = value_at_index >> 24; // for Map Insertion when it contain more values

        // Update the hashvalue or maintain the hashvalue for clusters.
        value_at_index = value_at_index | (1 << 1);

        // value_at_index=value_at_index|(1 << clusterID);
        *address_of_index = value_at_index;

        // Update map.
    }
    else
    {

        value_at_index |= (1 << 0); // bit masking to set on position 0 if the array position is 0
        value_at_index |= (cluster_id_ << 24);
        *address_of_index = value_at_index;
    }
}

std::string generateRandomString(int len)
{
    std::string str;
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < len; ++i)
    {
        str += alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    return str;
}

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
        std::string left_r, right_r, clinicalAreas, embedding;
        std::string skip_1, skip_2;

        // if(rowIndex==768423)
        //  std::cout << "Original Line::: " << line << std::endl;
        // getline(ss, skip_1, ',');
        // getline(ss, skip_2, ',');
        // getline(ss, embedding, ';');
        getline(ss, clinicalAreas, ';');
        //  getline(ss, embedding, ';');
        // Vector of char* to store each split string
        std::vector<char *> clinicalAreaParts;
        std::vector<std::string> clinicalareastring;
        std::stringstream areaStream(clinicalAreas);

        std::string part;
        // Split by comma and add each part to the vector
        while (std::getline(areaStream, part, ','))
        {
            // Trim whitespace from the part
            part.erase(0, part.find_first_not_of(" \t\n\r"));
            part.erase(part.find_last_not_of(" \t\n\r") + 1);
            clinicalareastring.push_back(part);
            char *cstr = strdup(part.c_str()); // Convert std::string to char*
            clinicalAreaParts.emplace_back(cstr);
        }

        hashTable[rowIndex] = clinicalAreaParts;

        rowIndex++;
    }
    // cout<<"RowIndex"<<rowIndex<<endl;
    // Close the file after reading

    // for (const auto &pair : hashTable) {
    //     unsigned int key = pair.first;
    //     const std::vector<char *> &values = pair.second;

    //     // Print the key
    //     std::cout << "Key: " << key << " -> Values: ";

    //     // Print the vector of char*
    //     for (const char *value : values) {
    //         std::cout << value << " ";  // Print the C-string
    //     }

    //     std::cout << std::endl;
    // }

    file.close();
    return hashTable;
}

/// Reading all from directory

bool loadCMSFiles(const std::string &folderPath, std::unordered_map<unsigned int, CountMinSketchMinHash> &mapForCMS)
{
    DIR *dir;
    struct dirent *entry;

    dir = opendir(folderPath.c_str());
    if (dir == nullptr)
    {
        std::cerr << "Failed to open directory: " << folderPath << std::endl;
        return false;
    }

    try
    {
        while ((entry = readdir(dir)) != nullptr)
        {
            std::string filename = entry->d_name;

            // Check if file ends with .bin
            if (filename.length() >= 4 &&
                filename.substr(filename.length() - 4) == ".bin")
            {

                std::string fullPath = folderPath + "/" + filename;

                // Extract key from filename
                // Assuming filename format is "key.bin"
                std::string keyStr = filename.substr(0, filename.length() - 4);
                unsigned int key;
                try
                {
                    key = std::stoul(keyStr);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Failed to extract key from filename: " << filename << std::endl;
                    continue;
                }

                // Load CMS from file
                CountMinSketchMinHash *cms = CountMinSketchMinHash::loadFromFile(fullPath);
                if (cms != nullptr)
                {
                    // Move the loaded CMS into the map
                    mapForCMS[key] = std::move(*cms);
                    delete cms; // Clean up the pointer after moving
                }
                else
                {
                    std::cerr << "Failed to load CMS from file: " << fullPath << std::endl;
                }
            }
        }
        closedir(dir);
        return !mapForCMS.empty(); // Return true if at least one file was loaded
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        closedir(dir);
        return false;
    }
}

std::map<std::string, std::vector<unsigned int>> reading_range_data(std::string file_path)
{
    std::map<std::string, std::vector<unsigned int>> orderedMap;
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return orderedMap;
    }

    std::string line;
    int rowIndex = 0;

    // Read the file line by line
    getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string id, dateCreated;

        getline(ss, id, ',');

        getline(ss, dateCreated, ',');
        unsigned int id_ = std::stoi(id);

        if (orderedMap.find(dateCreated) != orderedMap.end())
        {
            // If the key exists, push the value into the vector
            orderedMap[dateCreated].push_back(id_);
        }
        else
        {
            // If the key doesn't exist, create a new vector and insert the value
            orderedMap[dateCreated] = std::vector<unsigned int>{id_};
        }
    }
    // Close the file after reading
    file.close();
    return orderedMap;
}

/// Conversion of string to the date
std::chrono::system_clock::time_point stringToTimePoint(const std::string &date)
{
    std::tm tm = {};
    std::istringstream ss(date);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

std::map<unsigned int, Vertex<std::string> *> leafNodes()
{

    std::ifstream file("/home/u6059148/DataSet/date_meta_data_file.csv");
    std::string line;
    // Read the file line by line
    getline(file, line);
    Vertex<std::string> *head = nullptr;                    // Start of the linked list
    Vertex<std::string> *tail = nullptr;                    // End of the linked list (last node)
    Vertex<std::string> *prev_vertex = nullptr;             // Keeps track of the previous vertex
    std::map<unsigned int, Vertex<std::string> *> node_map; // Map to store nodes by their index

    while (getline(file, line))
    {

        std::stringstream ss(line);
        std::string index, date;
        getline(ss, index, ',');
        getline(ss, date, ',');

        // Create a new Vertex for each line
        Vertex<std::string> *current_vertex = new Vertex<std::string>();
        current_vertex->setCurrentNodePredicate(date);
        current_vertex->setNodeId(static_cast<unsigned int>(std::stoul(index)));

        // Link the nodes
        if (prev_vertex)
        {
            prev_vertex->setNext(current_vertex); // Set the next pointer of the previous node
            current_vertex->setPrev(prev_vertex); // Set the prev pointer of the current node
        }
        else
        {
            head = current_vertex; // First node becomes the head
        }

        prev_vertex = current_vertex; // Update prev_vertex to current
        tail = current_vertex;        // Update tail to the last node
        node_map[static_cast<unsigned int>(std::stoul(index))] = current_vertex;
    }
    file.close();
    return node_map;
}

std::unordered_map<unsigned int, std::pair<std::string, std::vector<char *>>> reading_metaData_hybrid_query(std::string file_path)
{
    std::unordered_map<unsigned int, std::pair<std::string, std::vector<char *>>> hashTable;
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return hashTable;
    }

    std::string line;
    int rowIndex = 0;

    // Read the file line by line
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::vector<char *> rowVector;

        size_t semicolonPos = line.find(';');
        if (semicolonPos == std::string::npos)
        {
            std::cerr << "Invalid format, no ';' found in line: " << line << std::endl;
            continue;
        }

        // Left part (before ';')
        std::string datePart = line.substr(0, semicolonPos);
        std::string areasPart = line.substr(semicolonPos + 1);

        // Split the right part by ',' and store in a vector of char*
        std::vector<char *> clinicalAreas;
        std::stringstream ss(areasPart);
        std::string area;

        while (std::getline(ss, area, ','))
        {
            char *areaCharPtr = new char[area.size() + 1];
            std::strcpy(areaCharPtr, area.c_str());
            clinicalAreas.push_back(areaCharPtr);
        }

        std::pair<std::string, std::vector<char *>> pair = {datePart, clinicalAreas};
        hashTable[rowIndex] = pair;

        rowIndex++;
    }
    // Close the file after reading
    file.close();
    return hashTable;
}

std::unordered_map<unsigned int, unsigned int> reading_metaData_int(std::string file_path)
{

    std::unordered_map<unsigned int, unsigned int> hashTable;
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
        std::stringstream ss(line);
        std::string qtext, dateCreated, clinicalAreas, embedding;
        // getline(ss, dateCreated, ';');
        // getline(ss, embedding, ';');
        getline(ss, clinicalAreas, ';');
        unsigned int views = 0;
        // cout<<clinicalAreas<<endl;
        views = std::stoul(clinicalAreas);

        // ss >> views; // Convert the first part directly into an integer

        // std::pair<std::string, std::vector<char *>> pair = {datePart, clinicalAreas};
        hashTable[rowIndex] = views;

        rowIndex++;
    }
    // cout<<"RowIndex"<<rowIndex<<endl;
    // Close the file after reading
    file.close();
    return hashTable;
}
