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

vector<std::vector<float>> reading_files_file(std::string file_path, bool flag_read);
std::unordered_map<unsigned int, std::vector<char *>> reading_metaData(std::string file_path);

bool loadCMSFiles(const std::string &folderPath, std::unordered_map<unsigned int, CountMinSketchMinHash> &mapForCMS);

std::map<std::string, std::vector<unsigned int>> reading_range_data(std::string file_path);
// Multi diemensional data
std::unordered_map<unsigned int, std::pair<std::string, std::vector<char *>>> reading_metaData_hybrid_query(std::string file_path);

// void saveToFile(const std::unordered_map<std::string, RangeSearch<std::string>> &searchMap, const std::string &filename);

// std::unordered_map<std::string, RangeSearch<std::string>> loadFromFile(const std::string &filename);

std::map<unsigned int, Vertex<std::string> *> leafNodes();

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

int main()
{

    int dim = 768; // Dimension of the elements

    int M = 16;              // Tightly connected with internal dimensionality of the data
                              // strongly affects the memory consumption
    int ef_construction =32; // Controls index search speed/build speed tradeoff 200
                              // Loading Data from file

    int max_elements = 0;

//   vector<std::vector<float>> data_vectors1= reading_files_file("/data4/hnsw/TripClick/documents_full.csv", true);

    
        

//         max_elements = data_vectors1.size(); // Maximum number of elements, should be known beforehand
//                                                       // // Initing index
//         cout<<"Max "<<max_elements<<endl;

//          std::ofstream file("/data3/""/Result_Trip_click_irange_768/datapoints.bin", std::ios::binary);
//     if (!file) {
//         std::cerr << "Failed to open file for writing!" << std::endl;
//         return 1;
//     }

//     // Write the number of points (n) and dimension (d) as 4 bytes each
//     file.write(reinterpret_cast<const char*>(&max_elements), sizeof(int));
//     file.write(reinterpret_cast<const char*>(&dim), sizeof(int));

//     // Write the data points (n * d * sizeof(float))
//     for (const auto& point : data_vectors1) {
//        // cout<<"Writing"<<endl;
//         file.write(reinterpret_cast<const char*>(point.data()), dim * sizeof(float));
//     }

//     file.close();                                              
//     exit(0);

 auto start = std::chrono::high_resolution_clock::now();
    pair<vector<std::vector<float>>, vector<vector<char *>>> data_vectors = reading_files("/data4/hnsw/TripClick/documents_full.csv", true);
       max_elements= data_vectors.first.size();
        std::cout << "Data has been Read successfully.   " <<max_elements<< std::endl;

        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        // // // cout<<max_elements<<endl;

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
            //for(int i=0;i<max_elements;i++){
              alg_hnsw->addPoint((void *)(data + dim * row), row);
           // }
              // alg_hnsw->addPointWithMetaData(data + (row * dim), row, data_vectors.second[row]);
               });

          alg_hnsw->saveIndex("/data4/hnsw/TripClick/IndexFile/16_32.bin");
      //  exit(0);
 auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double qps = static_cast<double>(duration) / 1000.0;
          std::cout << "Time taken for post_filtering in seconds: " << qps << std::endl;
    
    alg_hnsw->freeMemory();
    //  delete alg_hnsw;


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
    getline(file, line);
    int test_counter = 0;

    while (getline(file, line))
    {

        std::stringstream ss(line);
        std::string qtext, dateCreated, clinicalAreas, embedding;

        // Assuming columns are separated by semicolons
        // if (flag_read == true)
            // getline(ss, qtext, ';');
            // getline(ss, embedding, ';');
            getline(ss, dateCreated, ';');
       
            getline(ss, clinicalAreas, ';');
        
             getline(ss, embedding, ';');
        //   getline(ss, clinicalAreas, ';');
        // Initialize vectors

        vector<float> embeddingVector;
        vector<char *> clinicalAreasVector;

        if (!isNullOrEmpty(embedding)) // && !isNullOrEmpty(dateCreated)
        {
            // Split clinicalAreas and embedding
            embeddingVector = splitToFloat(embedding, ',');
           // clinicalAreasVector = splitString(dateCreated, ',');
            // Here do one thing

            if (embeddingVector.size() == 768) // 768 for clinical data
            {
              
                total_embeddings.push_back(embeddingVector);
               // predicates.push_back(clinicalAreasVector);
            }
        }
        test_counter++;
        if (test_counter % 100000 == 0)
        {
            cout << "Counter" << test_counter << endl;
        }
    }
    cout<< total_embeddings.size()<<endl;
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
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::vector<char *> rowVector;

        // Split the line by ';' and store each token into the vector
        std::vector<char *> tokens = splitString(line, ';');

        // Add each token (char*) into the row vector
        for (char *token : tokens)
        {
            rowVector.push_back(token);
        }

        // Insert into the hash table with rowIndex as the key
        hashTable[rowIndex] = rowVector;

        // Increment the row index for the next entry
        rowIndex++;
    }
    // Close the file after reading
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

// // Function to save the entire map to a file
// void saveToFile(const std::unordered_map<std::string, RangeSearch<std::string>> &searchMap, const std::string &filename)
// {
//     std::ofstream out(filename, std::ios::binary);
//     if (out.is_open())
//     {
//         size_t mapSize = searchMap.size();
//         out.write((char *)&mapSize, sizeof(mapSize));

//         for (const auto &[key, range] : searchMap)
//         {
//             size_t len = key.size();
//             out.write((char *)&len, sizeof(len));
//             out.write(key.c_str(), len);
//             range.save(out);
//         }

//         out.close();
//     }
// }
// // Function to load the map from a file
// std::unordered_map<std::string, RangeSearch<std::string>> loadFromFile(const std::string &filename)
// {
//     std::unordered_map<std::string, RangeSearch<std::string>> searchMap;
//     std::ifstream in(filename, std::ios::binary);
//     if (in.is_open())
//     {
//         size_t mapSize;
//         in.read((char *)&mapSize, sizeof(mapSize));

//         for (size_t i = 0; i < mapSize; ++i)
//         {
//             std::string key;
//             size_t len;
//             in.read((char *)&len, sizeof(len));
//             key.resize(len);
//             in.read(&key[0], len);

//             RangeSearch<std::string> range;
//             range.load(in);
//             searchMap[key] = range;
//         }

//         in.close();
//     }
//     return searchMap;
// }

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

vector<std::vector<float>> reading_files_file(std::string file_path, bool flag_read)
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
        std::string qtext, dateCreated, clinicalAreas, embedding;

        // Assuming columns are separated by semicolons
        // if (flag_read == true)
        //     getline(ss, qtext, ';');

        getline(ss, dateCreated, ';');
       // getline(ss, embedding, ';');
        getline(ss, clinicalAreas, ';');
        getline(ss, embedding, ';');
       
        //   getline(ss, clinicalAreas, ';');
        // Initialize vectors

        vector<float> embeddingVector;
        vector<char *> clinicalAreasVector;

        if (!isNullOrEmpty(embedding)) // && !isNullOrEmpty(dateCreated)
        {
            // Split clinicalAreas and embedding
            embeddingVector = splitToFloat(embedding, ',');
           // clinicalAreasVector = splitString(dateCreated, ',');
            // Here do one thing

            if (embeddingVector.size() == 768) // 768 for clinical data
            {
              // cout<<"I am here"<<endl;
                total_embeddings.push_back(embeddingVector);
               // predicates.push_back(clinicalAreasVector);
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
    std::vector<std::vector<float>> data_vectors(total_embeddings);
    return data_vectors;
}
