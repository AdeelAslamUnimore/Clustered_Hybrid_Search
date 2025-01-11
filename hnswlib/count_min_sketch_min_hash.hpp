
#pragma once
// define some constants
# define LONG_PRIME 4294967311l
# define MIN(a,b)  (a < b ? a : b)
#include "hnswlib/BPlusTree.h"
#include <vector>
#include <set>
#include <utility>
#include <fstream>
#include <iostream>



/** CountMinSketch class definition here **/
struct CountMinSketchMinHash {
  // width, depth 
  unsigned int w=1000,d=4;
  
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

  //MinHash vector
  BPlusTree<int>*** min_hash_keys;
//Total keys for minHash
  int keys=256;
  //Seed for Hash function
  unsigned int seed= 42 ;
  //Using Red Black Tree
  std::vector<std::vector<std::set<char>>> min_hash_RBT;
  // generate "new" aj,bj
  void genajbj(int **hashes, int i);

public:
  // constructor
  CountMinSketchMinHash();
  
  // update item (int) by count c
  void update(int item, int id, int c);
  // update item (string) by count c
  void update(const char *item, int id, int c);

  // estimate count of item i and return count
 std::pair<unsigned int,unsigned int>  estimate(int item) ;
  std::pair<unsigned int,unsigned int>estimate(const char *item);

  // return total count
  unsigned int totalcount();

  // generates a hash value for a string
  // same as djb2 hash function
  unsigned int hashstr(const char *str);
  //HashFunction for min_hash
  uint64_t MurmurHash64B ( const void * key, int len, unsigned int seed );
  //Writing the data
  bool saveToFile(const std::string& filename) ;
  //Loading the file
  static CountMinSketchMinHash* loadFromFile(const std::string& filename);

  // destructor
 // ~CountMinSketchMinHash();
};


