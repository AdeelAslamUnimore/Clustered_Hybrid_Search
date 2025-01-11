#include "hnswlib.h"
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

// B+ tree class with multiple values per key
template <typename K, typename V>
class BLinkTree
{
    struct Node
    {
        bool isLeaf;
        vector<K> keys;
        vector<Node *> children;
        Node *next;
        // Store values only in leaf nodes
        vector<vector<V>> values; // Each key can have multiple values

        Node(bool leaf = false)
            : isLeaf(leaf), next(nullptr)
        {
        }

        // Destructor for Node
        ~Node()
        {
            // Delete all children
            for (Node *child : children)
            {
                delete child; // This will recursively delete the subtree
            }
        }
    };

    Node *root;
    int t; // Minimum degree

    void splitChild(Node *parent, int index, Node *child);
    void insertNonFull(Node *node, K key, V value);
    void remove(Node *node, K key);
    void borrowFromPrev(Node *node, int index);
    void borrowFromNext(Node *node, int index);
    void merge(Node *node, int index);
    void printTree(Node *node, int level);
    /// Distance function

    hnswlib::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_{nullptr};

public:
    BLinkTree(int degree, hnswlib::SpaceInterface<float> *s)
        : root(nullptr),
          t(degree),
          fstdistfunc_(s->get_dist_func()),
          dist_func_param_(s->get_dist_func_param())
    {
        // Additional initialization code if needed
    }
    ~BLinkTree()
    {
        delete root;
    }

    void insert(K key, V value);
    vector<V> search(K key);
    void remove(K key);
    vector<pair<K, vector<V>>> rangeQuery(K lower, K upper, const char *data_level0_memory_, size_t offsetData_, size_t size_data_per_element_, const void *query_data, std::unordered_set<int> &visted_ids, std::vector<std::pair<int, float>> &result_vector);
    void printTree();
};

template <typename K, typename V>
void BLinkTree<K, V>::splitChild(Node *parent, int index, Node *child)
{
    Node *newChild = new Node(child->isLeaf);
    parent->children.insert(parent->children.begin() + index + 1, newChild);
    parent->keys.insert(parent->keys.begin() + index, child->keys[t - 1]);

    newChild->keys.assign(child->keys.begin() + t, child->keys.end());
    child->keys.resize(t - 1);

    if (child->isLeaf)
    {
        // Handle values in leaf nodes
        newChild->values.assign(child->values.begin() + t, child->values.end());
        child->values.resize(t - 1);
        newChild->next = child->next;
        child->next = newChild;
    }
    else
    {
        newChild->children.assign(child->children.begin() + t, child->children.end());
        child->children.resize(t);
    }
}

template <typename K, typename V>
void BLinkTree<K, V>::insertNonFull(Node *node, K key, V value)
{
    if (node->isLeaf)
    {
        auto it = upper_bound(node->keys.begin(), node->keys.end(), key);
        int pos = it - node->keys.begin();

        // Check if key already exists
        auto findIt = find(node->keys.begin(), node->keys.end(), key);
        if (findIt != node->keys.end())
        {
            int existingPos = findIt - node->keys.begin();
            node->values[existingPos].push_back(value);
        }
        else
        {
            node->keys.insert(it, key);
            node->values.insert(node->values.begin() + pos, vector<V>{value});
        }
    }
    else
    {
        int i = node->keys.size() - 1;
        while (i >= 0 && key < node->keys[i])
        {
            i--;
        }
        i++;
        if (node->children[i]->keys.size() == 2 * t - 1)
        {
            splitChild(node, i, node->children[i]);
            if (key > node->keys[i])
            {
                i++;
            }
        }
        insertNonFull(node->children[i], key, value);
    }
}

template <typename K, typename V>
vector<V> BLinkTree<K, V>::search(K key)
{
    Node *current = root;
    while (current != nullptr)
    {
        int i = 0;
        while (i < current->keys.size() && key > current->keys[i])
        {
            i++;
        }

        if (current->isLeaf)
        {
            if (i < current->keys.size() && key == current->keys[i])
            {
                return current->values[i];
            }
            return vector<V>(); // Return empty vector if key not found
        }

        if (i < current->keys.size() && key == current->keys[i])
        {
            current = current->children[i + 1];
        }
        else
        {
            current = current->children[i];
        }
    }
    return vector<V>();
}

template <typename K, typename V>
vector<pair<K, vector<V>>> BLinkTree<K, V>::rangeQuery(K lower, K upper, const char *data_level0_memory_, size_t offsetData_, size_t size_data_per_element_, const void *query_data, std::unordered_set<int> &visited_ids, std::vector<std::pair<int, float>> &result_vector)
{
    vector<pair<K, vector<V>>> result;
    if (!root)
        return result;

    Node *current = root;
    while (!current->isLeaf)
    {
        int i = 0;
        while (i < current->keys.size() && lower > current->keys[i])
        {
            i++;
        }
        current = current->children[i];
    }

    while (current != nullptr)
    {
        for (size_t i = 0; i < current->keys.size(); i++)
        {
            if (current->keys[i] >= lower && current->keys[i] <= upper)
            {
                for (size_t j = 0; j < current->values[i].size(); j++)
                {

                    unsigned int value= current->values[i][j];
                             if (visited_ids.count(value))
                            {
                                continue;
                            }
                    visited_ids.insert(value);
                    const char *getDataByInternalId1 = data_level0_memory_ + value * size_data_per_element_ + offsetData_;

                    float dist1 = fstdistfunc_(query_data, getDataByInternalId1, dist_func_param_);
                   result_vector.push_back(std::make_pair(value, dist1));
                }
                // c
                // std::cout<<"Distance"<<dist1<<std::endl;
                // result.push_back({current->keys[i], current->values[i]});
            }
            if (current->keys[i] > upper)
            {
                return result;
            }
        }
        current = current->next;
    }
    return result;
}

template <typename K, typename V>
void BLinkTree<K, V>::insert(K key, V value)
{
    if (root == nullptr)
    {
        root = new Node(true);
        root->keys.push_back(key);
        root->values.push_back(vector<V>{value});
    }
    else
    {
        if (root->keys.size() == 2 * t - 1)
        {
            Node *newRoot = new Node();
            newRoot->children.push_back(root);
            splitChild(newRoot, 0, root);
            root = newRoot;
        }
        insertNonFull(root, key, value);
    }
}
