#pragma once
#include <iostream>
#include <vector>
#include <memory>
template <typename T>
class Vertex
{
private:
    unsigned int node_id;
    T current_node_predicate;
    Vertex *prev_node;
    Vertex *next_node;

public:
    // Default constructor
    Vertex() : node_id(0), current_node_predicate(T()), prev_node(nullptr), next_node(nullptr)
    {
        //  std::cout << "Node created with default constructor" << std::endl;
    }

    // Destructor
    ~Vertex() {}
    

    // Getters
    unsigned int getNodeId() const
    {
        return node_id;
    }

    T getCurrentNodePredicate() const
    {
        return current_node_predicate;
    }

    Vertex *getNext() const
    {
        return next_node; // Return a raw pointer for access
    }

    Vertex *getPrev() const
    {
        return prev_node; // Return a raw pointer for access
    }

    // Setters
    void setNodeId(unsigned int id)
    {
        node_id = id;
    }

    void setCurrentNodePredicate(const T &predicate)
    {
        current_node_predicate = predicate;
    }

    void setNext(Vertex *next)
    {
        next_node = next;
    }

    void setPrev(Vertex *prev)
    {
        prev_node = prev;
    }

    // Additional methods to transfer ownership of next and previous nodes
    // std::unique_ptr<Vertex> releaseNext() {
    //     return std::move(next_node);  // Transfer ownership
    // }

    // std::unique_ptr<Vertex> releasePrev() {
    //     return std::move(prev_node);  // Transfer ownership
    // }
};