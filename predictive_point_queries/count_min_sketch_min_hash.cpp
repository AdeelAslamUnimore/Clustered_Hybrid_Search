#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <random>
#include "count_min_sketch_min_hash.h"
#include <utility>
#include <set>
#include <queue>
#include <algorithm>
#include <fstream>
using namespace std;
/** CountMinSketchMinHash constructor
 * Includes initialization of counters and hash functions
 * Used for predictive point queries
 * Also emplyed minHash for each counter
 */
CountMinSketchMinHash::CountMinSketchMinHash()
{
    C = new int *[d];
    unsigned int i, j;
    // This is just conisdering the fingureprint for eight bits laterly we can extend it to more bits
    full_keys = std::vector<std::vector<std::set<uint16_t>>>(d, std::vector<std::set<uint16_t>>(w)); // Employed only for holding the complete keys. // 16 bits
    // intilization for each counter in the count min sketch
    for (i = 0; i < d; i++)
    {
        C[i] = new int[w];
        for (j = 0; j < w; j++)
        {
            C[i][j] = 0;
        }
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
// compute the total count of items in CMS
unsigned int CountMinSketchMinHash::totalcount()
{
    return total;
}

// Update the count min sketch for integer item
void CountMinSketchMinHash::update_vector(int item, int id, int c)
{
    unsigned int hashval = 0;
    
    for (unsigned int j = 0; j < d; j++)
    {
        // -------------------------------
        // Count-Min Sketch update
        // -------------------------------
        hashval = (static_cast<long>(hashes[j][0]) * item) % LONG_PRIME;
        hashval = (hashval + hashes[j][1]) % LONG_PRIME; // second hash
        hashval = (hashval % w + w) % w;

        C[j][hashval] += c;

        //-------------------------------
        // Min-hash update with 16-bit fingerprint
        // -------------------------------
        uint64_t h = MurmurHash64B(&id, sizeof(id), seed);
        uint16_t fingerprint = static_cast<uint16_t>(h >> 48); // top 16 bits

        std:set<uint16_t> &s = full_keys[j][hashval];

        // Insert new fingerprint
        s.insert(fingerprint);

        // Ensure we only keep the smallest 'keys' elements
        if (s.size() > keys)
        {
            // Remove the largest element
            auto it = std::prev(s.end()); // last element
            s.erase(it);
        }
    }

    
}

// Update the count min sketch for string item
void CountMinSketchMinHash::update(const std::string &item, int id, int c)
{
    int hashval = hashstr(item.c_str());
    update_vector(hashval, id, c);
}

// CountMinSketch estimate item count (int)
// Return function only give me the index of the Count Min sketch
// Which later we employs for minHash extraction
// first pair is the row second is the column index in 2 D array
std::pair<unsigned int, unsigned int> CountMinSketchMinHash::estimate(int item)
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

        if (C[j][hashval] < minval)
        {

            minval = C[j][hashval];
            min_j = j;
            min_hashval = hashval;
        }
    }

    return std::make_pair(min_j, min_hashval);
}

// CountMinSketch estimate item count (string)
std::pair<unsigned int, unsigned int> CountMinSketchMinHash::estimate(const std::string &str)
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

// generates a hash value for a string (std::string version)
// same as djb2 hash function
unsigned int CountMinSketchMinHash::hashstr(const std::string &str)
{
    unsigned long hash = 5381;

    for (char c : str) // Range-based for loop
    {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }

    return hash;
}