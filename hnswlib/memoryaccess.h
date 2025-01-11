// data.h
#ifndef DATA_H   // Include guard to prevent multiple inclusions
#define DATA_H

#include <stdint.h>  // For uint8_t, uint32_t

// Union definition
union Data {
    struct {
        uint8_t bytes[3];
        uint8_t padding;
    } raw;
    uint32_t value;
};

#endif // DATA_H
