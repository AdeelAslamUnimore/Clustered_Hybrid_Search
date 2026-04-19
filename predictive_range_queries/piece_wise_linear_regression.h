#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <utility> // for std::pair
#include <sstream> // for std::istringstream
#include <ctime>
#include <unordered_map>
#include <utility>   // std::pair
#include <stdexcept> // std::invalid_argument, std::runtime_error
#include <algorithm> // std::sort, std::lower_bound
#include <cmath>
#include <iomanip>
#include <set> // std::max, std::min

// ---------------------------------------------------------------------------
// A single linear segment: covers keys in [x_start, x_end]
// and holds its own slope + intercept fitted by OLS on that slice.
// ---------------------------------------------------------------------------
struct Segment
{
    double x_start; // first key in this segment
    double x_end;   // last  key in this segment
    double slope;
    double intercept;
};

// ---------------------------------------------------------------------------
// PiecewiseRegressionModel
//
// Usage (integer keys, mirrors your RegressionModel::train_int style):
//
//   PiecewiseRegressionModel model;
//   model.setNumSegments(16);          // how many segments you want
//   model.train_int(cdf_vector);       // vector<pair<int,double>>
//   double p = model.predict_int(key); // returns value clamped to [0,1]
//
// Usage (double keys, mirrors RegressionModel::train / predict):
//
//   model.train(cdf_vector);           // vector<pair<double,double>>
//   double p = model.predict(key);
// ---------------------------------------------------------------------------
class PiecewiseRegressionModel
{
private:
    std::vector<Segment> segments; // fitted segments, sorted by x_start
    int num_segments;              // how many segments to split data into

    // -----------------------------------------------------------------------
    // Core OLS fit for one segment.
    // Returns false (and sets slope=0,intercept=mean_y) if all x are equal.
    // -----------------------------------------------------------------------
    std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;
    std::unordered_map<int, std::vector<uint16_t>> map_cdf_range_k_minwise_full;
    unsigned int total;

    bool fitSegment(const std::vector<std::pair<double, double>> &pts,
                    double &slope, double &intercept) const
    {
        size_t n = pts.size();
        if (n == 0)
            return false;

        if (n == 1)
        {
            // Only one point — horizontal line at that y value
            slope = 0.0;
            intercept = pts[0].second;
            return true;
        }

        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        for (const auto &p : pts)
        {
            sum_x += p.first;
            sum_y += p.second;
            sum_xx += p.first * p.first;
            sum_xy += p.first * p.second;
        }

        double denom = n * sum_xx - sum_x * sum_x;
        if (denom == 0.0)
        {
            // All x values identical — flat line at mean y
            slope = 0.0;
            intercept = sum_y / n;
            return false;
        }

        slope = (n * sum_xy - sum_x * sum_y) / denom;
        intercept = (sum_y - slope * sum_x) / n;
        return true;
    }

    // -----------------------------------------------------------------------
    // Internal: build segments from a sorted vector of (x, cdf) pairs.
    // -----------------------------------------------------------------------
    void buildSegments(std::vector<std::pair<double, double>> &sorted_data)
    {
        size_t n = sorted_data.size();
        if (n == 0)
            throw std::invalid_argument("Data is empty, cannot build segments.");

        segments.clear();

        // If fewer points than segments requested, use one segment per point
        int effective_segs = std::min((int)n, num_segments);
        size_t seg_size = n / effective_segs;  // points per segment
        size_t remainder = n % effective_segs; // leftover points

        size_t start = 0;
        for (int s = 0; s < effective_segs; ++s)
        {
            // Distribute remainder points across early segments
            size_t this_size = seg_size + (s < (int)remainder ? 1 : 0);
            size_t end = start + this_size; // exclusive

            // Slice for this segment
            std::vector<std::pair<double, double>> slice(
                sorted_data.begin() + start,
                sorted_data.begin() + end);

            Segment seg;
            seg.x_start = slice.front().first;
            seg.x_end = slice.back().first;
            fitSegment(slice, seg.slope, seg.intercept);
            segments.push_back(seg);

            start = end;
        }
    }

    // -----------------------------------------------------------------------
    // Find which segment a key falls into (binary search on x_end).
    // Returns the last segment if key > all x_end values.
    // -----------------------------------------------------------------------
    const Segment &findSegment(double x) const
    {
        // Linear scan is fine for small num_segments (<= 64).
        // For very large num_segments use binary search below instead.
        for (size_t i = 0; i < segments.size() - 1; ++i)
        {
            if (x <= segments[i].x_end)
                return segments[i];
        }
        return segments.back();
    }

public:
    // -----------------------------------------------------------------------
    // Constructor — default 8 segments (tune as needed)
    // -----------------------------------------------------------------------
    PiecewiseRegressionModel() : num_segments(1) {}

    // -----------------------------------------------------------------------
    // Set number of segments BEFORE calling train / train_int
    // -----------------------------------------------------------------------
    void setNumSegments(int n)
    {
        if (n < 1)
            throw std::invalid_argument("Number of segments must be >= 1.");
        num_segments = n;
    }

    int getNumSegments() const { return num_segments; }

    // -----------------------------------------------------------------------
    // Access fitted segments (e.g. for inspection / serialisation)
    // -----------------------------------------------------------------------
    const std::vector<Segment> &getSegments() const { return segments; }

    // -----------------------------------------------------------------------
    // train()  — double keys, matches your RegressionModel::train() style
    // Expects: vector<pair<double, double>> where .first = key, .second = CDF
    // Data does NOT need to be pre-sorted — we sort internally.
    // -----------------------------------------------------------------------
    void train(std::vector<std::pair<double, double>> &cdf)
    {
        if (cdf.size() < 2)
            throw std::invalid_argument("Need at least 2 data points to train.");

        // Sort by key (x) ascending — CDF must be monotone
        std::sort(cdf.begin(), cdf.end(),
                  [](const std::pair<double, double> &a,
                     const std::pair<double, double> &b)
                  { return a.first < b.first; });

        buildSegments(cdf);
    }

    // -----------------------------------------------------------------------
    // train_int()  — integer keys, mirrors your RegressionModel::train_int()
    // Expects: vector<pair<int, double>> where .first = key, .second = CDF
    // -----------------------------------------------------------------------
    void train_int(std::vector<std::pair<int, double>> &cdf)
    {
        if (cdf.size() < 2)
            throw std::invalid_argument("Need at least 2 data points to train.");

        // Convert to double pairs for unified internal processing
        std::vector<std::pair<double, double>> data;
        data.reserve(cdf.size());
        for (const auto &p : cdf)
            data.emplace_back(static_cast<double>(p.first), p.second);

        buildSegments(data);
    }

    // -----------------------------------------------------------------------
    // predict()  — double key, returns CDF estimate clamped to [0, 1]
    // -----------------------------------------------------------------------
    double predict(double key) const
    {
        if (segments.empty())
            throw std::runtime_error("Model has not been trained yet.");

        const Segment &seg = findSegment(key);
        double val = seg.slope * key + seg.intercept;
        return std::max(0.0, std::min(1.0, val));
    }

    // -----------------------------------------------------------------------
    // predict_int()  — integer key, mirrors your RegressionModel::predict_int()
    // -----------------------------------------------------------------------
    double predict_int(const int &key) const
    {
        return predict(static_cast<double>(key));
    }

    // -----------------------------------------------------------------------
    // Utility: print all segments (useful for debugging)
    // -----------------------------------------------------------------------
    void printSegments() const
    {
        std::cout << "PiecewiseRegressionModel — " << segments.size()
                  << " segment(s):\n";
        for (size_t i = 0; i < segments.size(); ++i)
        {
            std::cout << "  [" << i << "]"
                      << "  x_range=[" << segments[i].x_start
                      << ", " << segments[i].x_end << "]"
                      << "  slope=" << segments[i].slope
                      << "  intercept=" << segments[i].intercept
                      << "\n";
        }
    }

    // Function to convert date string to time_t (Unix timestamp)
    time_t convertDateStringToTimestamp(const std::string &dateStr)
    {
        std::tm tm = {};
        std::istringstream ss(dateStr);

        // Assuming the date format is YYYY-MM-DD

        ss >> std::get_time(&tm, "%Y-%m-%d");
        if (ss.fail())
        {

            throw std::invalid_argument("Date format is incorrect.");
        }

        // Convert to time_t
        return std::mktime(&tm);
    }

    // Getter for map_cdf_range_k_minwise (returns a const reference)
    const std::unordered_map<int, std::set<char>> &getMapCdfRangeKMinwise() const
    {
        return map_cdf_range_k_minwise;
    }

    // Setter for map_cdf_range_k_minwise (takes a const reference)
    void setMapCdfRangeKMinwise(const std::unordered_map<int, std::set<char>> &new_map)
    {
        map_cdf_range_k_minwise = new_map;
    }

    // Getter for full keys
    const std::unordered_map<int, std::vector<uint16_t>> &getMapCdfRangeKMinwiseFull() const
    {
        return map_cdf_range_k_minwise_full;
    }

    // Setter for full keys
    void setMapCdfRangeKMinwiseFull(const std::unordered_map<int, std::vector<uint16_t>> &new_map)
    {
        map_cdf_range_k_minwise_full = new_map;
    }

    // Getter for total count
    unsigned int getTotal() const
    {
        return total;
    }

    // Setter for total count
    void setTotal(unsigned int new_total)
    {
        total = new_total;
    }

    // Destructor
    ~PiecewiseRegressionModel() = default;
};