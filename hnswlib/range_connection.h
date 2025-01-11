
#pragma once
#include <iostream>
#include <vector>
#include <string>

template <typename T>
class RangeSearch {
private:
    T range_data;
    T prev_range_data;
    T next_range_data;
    std::vector<int> curr_nodes;
    // std::vector<int> prev_nodes;
    // std::vector<int> next_nodes;

public:
    // Default constructor
    RangeSearch() = default;

    // Destructor
    ~RangeSearch() = default;

    // Set the range data
    void setRangeData(T data) {
        range_data = data;
    }

    // Get the range data
    T getRangeData() const{
        return range_data;
    }

    // Set the previous node data
    void set_prev_node_data(T prev_node_data) {
        prev_range_data = prev_node_data;
    }

    // Get the previous node data
    T get_prev_node_data() const{
        return prev_range_data;
    }

    // Set the next node data
    void set_next_node_data(T next_node_data) {
        next_range_data = next_node_data;
    }

    // Get the next node data
    T get_next_node_data() const {
        return next_range_data;
    }

    // // Set previous nodes
    // void set_prev_nodes(const std::vector<unsigned int>& prev_nodes_vec) {
    //     prev_nodes.assign(prev_nodes_vec.begin(), prev_nodes_vec.end());
    // }

    // // Get previous nodes
    // const std::vector<int>& getPrevNodes() const {
    //     return prev_nodes;
    // }

    // Set next nodes
    void set_curr_nodes(const std::vector<unsigned int>& curr_nodes_vec) {
        curr_nodes.assign(curr_nodes_vec.begin(), curr_nodes_vec.end());
    }

    // Get next nodes
    const std::vector<int>& getCurrNodes() const {
        return curr_nodes;
    }

    // Save RangeSearch data to a binary file
    void save(std::ofstream& out) const {
        size_t len = range_data.size();
        out.write((char*)&len, sizeof(len));
        out.write(range_data.c_str(), len);
        
        len = prev_range_data.size();
        out.write((char*)&len, sizeof(len));
        out.write(prev_range_data.c_str(), len);
        
        len = next_range_data.size();
        out.write((char*)&len, sizeof(len));
        out.write(next_range_data.c_str(), len);
        
        len = curr_nodes.size();
        out.write((char*)&len, sizeof(len));
        out.write((char*)curr_nodes.data(), len * sizeof(int));
    }

 // Load RangeSearch data from a binary file
    void load(std::ifstream& in) {
        size_t len;
        
        in.read((char*)&len, sizeof(len));
        range_data.resize(len);
        in.read(&range_data[0], len);
        
        in.read((char*)&len, sizeof(len));
        prev_range_data.resize(len);
        in.read(&prev_range_data[0], len);
        
        in.read((char*)&len, sizeof(len));
        next_range_data.resize(len);
        in.read(&next_range_data[0], len);
        
        in.read((char*)&len, sizeof(len));
        curr_nodes.resize(len);
        in.read((char*)curr_nodes.data(), len * sizeof(int));
    }
    
};
