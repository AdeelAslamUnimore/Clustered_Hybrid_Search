// Ranges.h

#pragma once
#include <utility>
#include <fstream>
#include <iostream>
#ifndef RANGES_H
#define RANGES_H

// Defining range boundaries and labels
const int SIZE_Of_K_Min_WISE_HASH =256;
const unsigned int SEED= 42 ;
#define RANGE_0_0_1_START 0.0
#define RANGE_0_0_1_END 0.1
#define RANGE_0_0_1_LABEL 0

#define RANGE_0_1_2_START 0.1
#define RANGE_0_1_2_END 0.2
#define RANGE_0_1_2_LABEL 1

#define RANGE_0_2_3_START 0.2
#define RANGE_0_2_3_END 0.3
#define RANGE_0_2_3_LABEL 2

#define RANGE_0_3_4_START 0.3
#define RANGE_0_3_4_END 0.4
#define RANGE_0_3_4_LABEL 3

#define RANGE_0_4_5_START 0.4
#define RANGE_0_4_5_END 0.5
#define RANGE_0_4_5_LABEL 4

#define RANGE_0_5_6_START 0.5
#define RANGE_0_5_6_END 0.6
#define RANGE_0_5_6_LABEL 5

#define RANGE_0_6_7_START 0.6
#define RANGE_0_6_7_END 0.7
#define RANGE_0_6_7_LABEL 6

#define RANGE_0_7_8_START 0.7
#define RANGE_0_7_8_END 0.8
#define RANGE_0_7_8_LABEL 7

#define RANGE_0_8_9_START 0.8
#define RANGE_0_8_9_END 0.9
#define RANGE_0_8_9_LABEL 8

#define RANGE_0_9_10_START 0.9
#define RANGE_0_9_10_END 1.0
#define RANGE_0_9_10_LABEL 9

#define RANGE_0_10_11_START 1.0
#define RANGE_0_10_11_LABEL 10

// Function to check if a value is within a range
int check_range(double value)
{

    //return std::min((int) (value*10),10);
    // Good option so look it later for com
    if (value >= RANGE_0_0_1_START && value < RANGE_0_0_1_END)
    {
        return RANGE_0_0_1_LABEL;
    }
    else if (value >= RANGE_0_1_2_START && value < RANGE_0_1_2_END)
    {
        return RANGE_0_1_2_LABEL;
    }
    else if (value >= RANGE_0_2_3_START && value < RANGE_0_2_3_END)
    {
        return RANGE_0_2_3_LABEL;
    }
    else if (value >= RANGE_0_3_4_START && value < RANGE_0_3_4_END)
    {
        return RANGE_0_3_4_LABEL;
    }
    else if (value >= RANGE_0_4_5_START && value < RANGE_0_4_5_END)
    {
        return RANGE_0_4_5_LABEL;
    }
    else if (value >= RANGE_0_5_6_START && value < RANGE_0_5_6_END)
    {
        return RANGE_0_5_6_LABEL;
    }
    else if (value >= RANGE_0_6_7_START && value < RANGE_0_6_7_END)
    {
        return RANGE_0_6_7_LABEL;
    }
    else if (value >= RANGE_0_7_8_START && value < RANGE_0_7_8_END)
    {
        return RANGE_0_7_8_LABEL;
    }
    else if (value >= RANGE_0_8_9_START && value < RANGE_0_8_9_END)
    {
        return RANGE_0_8_9_LABEL;
    }
    else if (value >= RANGE_0_9_10_START && value <= RANGE_0_9_10_END)
    {
        return RANGE_0_9_10_LABEL;
    }
    else if (value >RANGE_0_10_11_START)
    {
        return RANGE_0_10_11_LABEL;
    }

    else
    {
        return -1; // Value doesn't match any range
    }
}
int check_range_search(double value)
{
return std::min((int) (value*10),10);
}

uint64_t MurmurHash64B(const void *key, int len, unsigned int seed)
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

#endif
