#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <utility> // for std::pair
#include <sstream> // for std::istringstream
#include <ctime>
#include <set>
#include <iomanip> // for std::tm and std::time_t
class RegressionModel
{
private:
    double slope;
    double intercept;
    std::unordered_map<int, std::set<char>> map_cdf_range_k_minwise;

public:
    // Default constructor
    RegressionModel() : slope(0.0), intercept(0.0) {}

    // Training The Model
    // Set the map
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

    void train(std::vector<std::pair<std::string, double>> &cdf)
    {
        // Check if the input vector is empty
        if (cdf.empty())
        {
            throw std::invalid_argument("Input vector must be non-empty.");
        }

        // Initialize sums for regression calculations
        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        size_t n = cdf.size(); // Store the size of the input vector

        // Iterate through the CDF vector
        for (size_t i = 0; i < n; ++i)
        {
            // Convert date string to timestamp (double)
            double dateAsDouble = static_cast<double>(convertDateStringToTimestamp(cdf[i].first));
            sum_x += dateAsDouble;                  // Update sum of x
            sum_y += cdf[i].second;                 // Update sum of y
            sum_xx += dateAsDouble * dateAsDouble;  // Update sum of x^2
            sum_xy += dateAsDouble * cdf[i].second; // Update sum of x*y
        }

        // Calculate the denominator for slope
        double denominator = n * sum_xx - sum_x * sum_x;

        // Check for division by zero
        if (denominator == 0)
        {
            throw std::runtime_error("Denominator in slope calculation is zero, cannot compute slope.");
        }

        // Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / denominator;
        intercept = (sum_y - slope * sum_x) / n;
    }

    void train_int(std::vector<std::pair<unsigned int, double>> &cdf)
    {
        // Check if the input vector is empty
        if (cdf.empty())
        {
            throw std::invalid_argument("Input vector must be non-empty.");
        }

        // Check if there's enough data for regression
        size_t n = cdf.size(); // Store the size of the input vector
        if (n < 2)
        {
            throw std::invalid_argument("At least two data points are required for linear regression.");
        }

        // Initialize sums for regression calculations
        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;

        // Iterate through the CDF vector
        for (size_t i = 0; i < n; ++i)
        {
            // Convert date string to timestamp (double)
            int item = cdf[i].first;
            sum_x += item;                  // Update sum of x
            sum_y += cdf[i].second;         // Update sum of y
            sum_xx += item * item;          // Update sum of x^2
            sum_xy += item * cdf[i].second; // Update sum of x*y
        }

        // Calculate the denominator for slope
        double denominator = n * sum_xx - sum_x * sum_x;

        // Check for division by zero
        if (denominator == 0)
        {
            throw std::runtime_error("Denominator in slope calculation is zero, cannot compute slope.");
        }

        // Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / denominator;
        intercept = (sum_y - slope * sum_x) / n;
    }

    // Predict the CDF.
    double predict(double &date)
    {
        // double x = static_cast<double>(convertDateStringToTimestamp(date));
        return slope * date + intercept;
    }

    double predict_int(unsigned int &item) const
    {
        double predicted_value = slope * item + intercept;
        // Apply sigmoid to the predicted value to ensure it's between 0 and 1
        return 1.0 / (1.0 + std::exp(-predicted_value));
    }
    // Save the model
    void save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        if (!ofs)
        {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        ofs << slope << "\n"
            << intercept;
        ofs.close();
    }
    // Load it back
    void load(const std::string &filename)
    {
        std::ifstream ifs(filename);
        if (!ifs)
        {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        ifs >> slope >> intercept;
        ifs.close();
    }

    // Function to convert date string to time_t (Unix timestamp)
    // Function to convert date string to time_t (Unix timestamp)
    time_t convertDateStringToTimestamp(const std::string &dateStr)
    {
        std::tm tm = {};
        std::istringstream ss(dateStr);

        // Assuming the date format is YYYY-MM-DD

        ss >> std::get_time(&tm, "%Y-%m-%d");
        if (ss.fail())
        {
            cout << "Date is" << dateStr << endl;
            throw std::invalid_argument("Date format is incorrect.");
        }

        // Convert to time_t
        return std::mktime(&tm);
    }

    // Destructor
    ~RegressionModel() = default;
};
