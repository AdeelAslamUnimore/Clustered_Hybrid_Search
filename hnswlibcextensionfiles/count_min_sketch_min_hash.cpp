#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <random>
#include "hnswlib/count_min_sketch_min_hash.hpp"
#include "hnswlib/BPlusTree.h"
#include <utility>
#include <set>
#include <fstream>



using namespace std;

/**
   Class definition for CountMinSketch.
   public operations:
   // overloaded updates
   void update(int item, int c);
   void update(char *item, int c);
   // overloaded estimates
   unsigned int estimate(int item);
   unsigned int estimate(char *item);
**/


CountMinSketchMinHash::CountMinSketchMinHash()
{


  C = new int *[d];
 
  
  // allocating memory to min_hash_array
  //min_hash_keys = new BPlusTree<int> **[d];
  unsigned int i, j;
  min_hash_RBT = std::vector<std::vector<std::set<char>>>(d, std::vector<std::set<char>>(w));
  //  
  for (i = 0; i < d; i++)
  {
   
    C[i] = new int[w];
    // min_hash_keys[i] = new BPlusTree<int> *[w]; // Allocate each row for min_hash_keys
    for (j = 0; j < w; j++)
    {
      C[i][j] = 0;
      // min_hash_keys[i][j] = new BPlusTree<int>(8);
      // min_hash_keys[i][j]->insert(10);
    }
    // declaring the array for min_hashes
  }
  

  // initialize d pairwise independent hashes
  srand(time(NULL));
  hashes = new int *[d];
  for (i = 0; i < d; i++)
  {
    hashes[i] = new int[2];
    genajbj(hashes, i);
  }
  
}

// CountMinSkectch destructor
// CountMinSketchMinHash::~CountMinSketchMinHash()
// {
//   // free array of counters, C
//   cout<<"HereRemoving";
//   unsigned int i;
//   for (i = 0; i < d; i++)
//   {
//     delete[] C[i];
//   }
//   delete[] C;

//   // free array of hash values
//   for (i = 0; i < d; i++)
//   {
//     delete[] hashes[i];
//   }
//   delete[] hashes;
// }

// CountMinSketch totalcount returns the
// total count of all items in the sketch
unsigned int CountMinSketchMinHash::totalcount()
{
  return total;
}

// countMinSketch update item count (int)
void CountMinSketchMinHash::update(int item, int id, int c)
{

  //total = total + c;
  unsigned int hashval = 0;
 
  for (unsigned int j = 0; j < d; j++)
  {

    hashval = (static_cast<long>(hashes[j][0]) * item) % LONG_PRIME;
    hashval = (hashval + hashes[j][1]) % LONG_PRIME; // Add the second hash component
    hashval = (hashval % w + w) % w;
    C[j][hashval] = C[j][hashval] + c;

    // Valu1e
    //Perform  some operation on
    uint64_t hash_value_of_data = MurmurHash64B(static_cast<const void *>(&id), 25, seed);
    uint64_t least_significant_16_bits = hash_value_of_data & 0xFF;

    // Convert to short and ccndition of confining its limit
     char least_significant_bits = static_cast<char>(least_significant_16_bits);
   // cout<<"Min Hash Size"<< min_hash_RBT[j][hashval].size()<<endl;
    min_hash_RBT[j][hashval].insert(least_significant_bits);
    // min_hash_RBT[j][hashval].insert(id);
    int size_min_hash = min_hash_RBT[j][hashval].size();
    if (size_min_hash > keys)
    {
      auto it = std::prev(min_hash_RBT[j][hashval].end()); // Get iterator to last element it is sorted so the last one has greater hash value for 8 bits 
      // Remove the last element
      min_hash_RBT[j][hashval].erase(it);
     // exit(0);
    }
  }
}

// countMinSketch update item count (string)
void CountMinSketchMinHash::update(const char *str, int id, int c)
{
  int hashval = hashstr(str);
  update(hashval, id, c);
}

// CountMinSketch estimate item count (int)
 std::pair<unsigned int,unsigned int> CountMinSketchMinHash::estimate(int item) 
{
  int minval = std::numeric_limits<int>::max();
  unsigned int hashval = 0;
  unsigned int min_j = 0;       // Track the j value of the minimum
  unsigned int min_hashval = 0; // Track the hashval of the minimum
  
  for (unsigned int j = 0; j < d; j++)
  {
    // Calculate hash value

    hashval = (static_cast<long>(hashes[j][0]) * item) % LONG_PRIME;
    hashval = (hashval + hashes[j][1]) % LONG_PRIME; // Add the second hash component
    hashval = (hashval % w + w) % w;
 
    // Update the minimum value and track j and hashval
   
    if (C[j][hashval] <minval)
    {
      
      minval =C[j][hashval];
      min_j = j;
      min_hashval = hashval;
   
    }
  }

  //  cout<<"MinVal"<<min_hash_RBT[min_j][min_hashval].size()<<endl;
  // std::pair<unsigned int, std::set<char>> myPair = std::make_pair(minval, min_hash_RBT[min_j][min_hashval]);

  return std::make_pair(min_j, min_hashval);;
}

// CountMinSketch estimate item count (string)
std::pair<unsigned int,unsigned int> CountMinSketchMinHash::estimate(const char *str)  
{
  int hashval = hashstr(str);
  return estimate(hashval);
}

// generates aj,bj from field Z_p for use in hashing
void CountMinSketchMinHash::genajbj(int **hashes, int i)
{
  std::random_device rd;
  std::mt19937 gen(rd()); // Mersenne Twister random number generator

  // Define a uniform distribution in the range [1, LONG_PRIME]
  std::uniform_int_distribution<long> dist(1, LONG_PRIME);
  hashes[i][0] = dist(gen); // Independent value for first component
  hashes[i][1] = dist(gen);
}

// generates a hash value for a sting
// same as djb2 hash function
unsigned int CountMinSketchMinHash::hashstr(const char *str)
{
  unsigned long hash = 5381;
  int c;
  while (c = *str++)
  {
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
  }
  return hash;
}

bool CountMinSketchMinHash::saveToFile(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        return false;
    }

    // Write simple types
    outFile.write(reinterpret_cast<const char*>(&w), sizeof(w));
    outFile.write(reinterpret_cast<const char*>(&d), sizeof(d));
    outFile.write(reinterpret_cast<const char*>(&eps), sizeof(eps));
    outFile.write(reinterpret_cast<const char*>(&gamma), sizeof(gamma));
    outFile.write(reinterpret_cast<const char*>(&aj), sizeof(aj));
    outFile.write(reinterpret_cast<const char*>(&bj), sizeof(bj));
    outFile.write(reinterpret_cast<const char*>(&total), sizeof(total));
    outFile.write(reinterpret_cast<const char*>(&keys), sizeof(keys));
    outFile.write(reinterpret_cast<const char*>(&seed), sizeof(seed));

    // Write array of arrays (C)
    for (unsigned int i = 0; i < d; ++i) {
        outFile.write(reinterpret_cast<const char*>(C[i]), w * sizeof(int));
    }

    // Write the 2D hashes array (hashes)
    for (unsigned int i = 0; i < d; ++i) {
        outFile.write(reinterpret_cast<const char*>(hashes[i]), 2 * sizeof(int));  // Assuming each row has 2 integers (aj, bj)
    }

    // // Write MinHash vector (min_hash_RBT)
    // for (const auto& vec : min_hash_RBT) {
    //     unsigned int vec_size = vec.size();
    //     outFile.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
    //     for (const auto& set : vec) {
    //         unsigned int set_size = set.size();
    //         outFile.write(reinterpret_cast<const char*>(&set_size), sizeof(set_size));
    //         for (const auto& ch : set) {
    //             outFile.write(&ch, sizeof(ch));
    //         }
    //     }
    // }

    // // Close file
     outFile.close();
    return true;
}


 CountMinSketchMinHash* CountMinSketchMinHash::loadFromFile(const std::string& filename) {
//     std::ifstream inFile(filename, std::ios::binary);
//     if (!inFile) {
//         return nullptr;
//     }

//     // Allocate new object
     CountMinSketchMinHash* cms = new CountMinSketchMinHash();

//     // Read simple types
//     inFile.read(reinterpret_cast<char*>(&cms->w), sizeof(cms->w));
//     inFile.read(reinterpret_cast<char*>(&cms->d), sizeof(cms->d));
//     inFile.read(reinterpret_cast<char*>(&cms->eps), sizeof(cms->eps));
//     inFile.read(reinterpret_cast<char*>(&cms->gamma), sizeof(cms->gamma));
//     inFile.read(reinterpret_cast<char*>(&cms->aj), sizeof(cms->aj));
//     inFile.read(reinterpret_cast<char*>(&cms->bj), sizeof(cms->bj));
//     inFile.read(reinterpret_cast<char*>(&cms->total), sizeof(cms->total));
//     inFile.read(reinterpret_cast<char*>(&cms->keys), sizeof(cms->keys));
//     inFile.read(reinterpret_cast<char*>(&cms->seed), sizeof(cms->seed));
   
//     // Allocate space for C array
//     cms->C = new int*[cms->d];
//     for (unsigned int i = 0; i < cms->d; ++i) {
//         cms->C[i] = new int[cms->w];
//         inFile.read(reinterpret_cast<char*>(cms->C[i]), cms->w * sizeof(int));
//     }


//  cms->hashes = new int*[cms->d];
//     for (unsigned int i = 0; i < cms->d; ++i) {
//         cms->hashes[i] = new int[2];  // Assuming each row contains two integers (aj, bj)
//         inFile.read(reinterpret_cast<char*>(cms->hashes[i]), 2 * sizeof(int));  // Read each row of `hashes`
//     }

//     // Read MinHash vector (min_hash_RBT)
//     for (unsigned int i = 0; i < cms->min_hash_RBT.size(); ++i) {
//         unsigned int vec_size;
//         inFile.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
//         std::vector<std::set<char>> vec(vec_size);
//         for (unsigned int j = 0; j < vec_size; ++j) {
//             unsigned int set_size;
//             inFile.read(reinterpret_cast<char*>(&set_size), sizeof(set_size));
//             std::set<char> charSet;
//             for (unsigned int k = 0; k < set_size; ++k) {
//                 char ch;
//                 inFile.read(&ch, sizeof(ch));
//                 charSet.insert(ch);
//             }
//             vec[j] = charSet;
//         }
//         cms->min_hash_RBT[i] = vec;
//     }

//     // Close file
//     inFile.close();
    return cms;
}






 

uint64_t CountMinSketchMinHash::MurmurHash64B(const void *key, int len, unsigned int seed)
{
  const unsigned int m = 0x5bd1e995;
  const int r = 24;

  unsigned int h1 = seed ^ len;
  unsigned int h2 = 0;

  const unsigned int *data = (const unsigned int *)key;

  while (len >= 8)
  {
    unsigned int k1 = *data++;
    k1 *= m;
    k1 ^= k1 >> r;
    k1 *= m;
    h1 *= m;
    h1 ^= k1;
    len -= 4;

    unsigned int k2 = *data++;
    k2 *= m;
    k2 ^= k2 >> r;
    k2 *= m;
    h2 *= m;
    h2 ^= k2;
    len -= 4;
  }

  if (len >= 4)
  {
    unsigned int k1 = *data++;
    k1 *= m;
    k1 ^= k1 >> r;
    k1 *= m;
    h1 *= m;
    h1 ^= k1;
    len -= 4;
  }

  switch (len)
  {
  case 3:
    h2 ^= ((unsigned char *)data)[2] << 16;
  case 2:
    h2 ^= ((unsigned char *)data)[1] << 8;
  case 1:
    h2 ^= ((unsigned char *)data)[0];
    h2 *= m;
  };

  h1 ^= h2 >> 18;
  h1 *= m;
  h2 ^= h1 >> 22;
  h2 *= m;
  h1 ^= h2 >> 17;
  h1 *= m;
  h2 ^= h1 >> 19;
  h2 *= m;

  uint64_t h = h1;

  h = (h << 32) | h2;

  return h;
}
