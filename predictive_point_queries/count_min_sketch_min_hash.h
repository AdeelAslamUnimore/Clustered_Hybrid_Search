#pragma once
// define some constants
#define LONG_PRIME 4294967311l
#define MIN(a, b) (a < b ? a : b)
#include <vector>
#include <stdint.h>
#include <set>
#include <utility>
#include <fstream>
#include <unordered_set>

#include <iostream>

/** CountMinSketch class definition here **/
struct CountMinSketchMinHash
{
  // width, depth
  unsigned int w = 1000, d = 4;

  // eps (for error), 0.01 < eps < 1
  // the smaller the better
  float eps;

  // gamma (probability for accuracy), 0 < gamma < 1
  // the bigger the better
  float gamma;

  // aj, bj \in Z_p
  // both elements of fild Z_p used in generation of hash
  // function
  unsigned int aj, bj;

  // total count so far
  unsigned int total;

  // array of arrays of counters
  int **C;

  // array of hash values for a particular item
  // contains two element arrays {aj,bj}
  int **hashes;

  // Total keys for minHash
  int keys = 1000;
  // Seed for Hash function
  unsigned int seed = 42;
  // To hold complete keys
  std::vector<std::vector<std::set<uint16_t>>> full_keys;

  void genajbj(int **hashes, int i);

public:
  // constructor
  CountMinSketchMinHash();

  // update item (int) by count c
  void update(int item, int id, int c);
  void update_vector(int item, int id, int c);
  // update item (string) by count c
  void update(const std::string &item, int id, int c);

  // estimate count of item i and return count
  std::pair<unsigned int, unsigned int> estimate(int item);
  std::pair<unsigned int, unsigned int> estimate(const std::string &item);

  // return total count
  unsigned int totalcount();

  // generates a hash value for a string
  // same as djb2 hash function
  unsigned int hashstr(const std::string &str);
  // HashFunction for min_hash
  uint64_t MurmurHash64B(const void *key, int len, unsigned int seed);
  // Writing the data
  bool saveToFile(const std::string &filename);
  // Loading the file
  static CountMinSketchMinHash *loadFromFile(const std::string &filename);

  // destructor
  // ~CountMinSketchMinHash();
};
