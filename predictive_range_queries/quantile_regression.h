#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <utility>   // std::pair
#include <sstream>   // std::istringstream
#include <ctime>
#include <unordered_map>
#include <stdexcept> // std::invalid_argument, std::runtime_error
#include <algorithm> // std::sort
#include <cmath>
#include <iomanip>
#include <set>

// ---------------------------------------------------------------------------
// QuantileRegressionModel
//
// Fits a single linear quantile regression:
//   Q_tau(y | x) = slope * x + intercept
// by minimising the pinball (check) loss via Iteratively Reweighted Least
// Squares (IRLS), which is numerically stable and avoids a linear-programming
// solver.
//
// Usage (integer keys, mirrors RegressionModel::train_int style):
//
//   QuantileRegressionModel model;
//   model.setTau(0.5);              // quantile to fit  (default 0.5 = median)
//   model.setMaxIter(200);          // IRLS iterations  (default 200)
//   model.setEpsilon(1e-6);         // IRLS weight floor (default 1e-6)
//   model.train_int(cdf_vector);    // vector<pair<int,double>>
//   double p = model.predict_int(key); // returns value clamped to [0,1]
//
// Usage (double keys, mirrors RegressionModel::train / predict):
//
//   model.train(cdf_vector);        // vector<pair<double,double>>
//   double p = model.predict(key);
// ---------------------------------------------------------------------------
class QuantileRegressionModel
{
private:
    // -----------------------------------------------------------------------
    // Fitted coefficients
    // -----------------------------------------------------------------------
    double slope;
    double intercept;
    bool   trained;

    // -----------------------------------------------------------------------
    // Hyper-parameters
    // -----------------------------------------------------------------------
    double       tau;      // target quantile in (0, 1)
    int          max_iter; // maximum IRLS iterations
    double       epsilon;  // small floor added to |residual| to avoid div-by-0

    // -----------------------------------------------------------------------
    // Mirrors of the extra members present in PiecewiseRegressionModel
    // (kept so the two classes are drop-in compatible)
    // -----------------------------------------------------------------------
    std::unordered_map<int, std::set<char>>        map_cdf_range_k_minwise;
    std::unordered_map<int, std::vector<uint16_t>> map_cdf_range_k_minwise_full;
    unsigned int total;

    // -----------------------------------------------------------------------
    // Weighted OLS helper used by IRLS.
    // Solves:  min_{\beta} sum_i w_i (y_i - slope*x_i - intercept)^2
    // Returns false if the system is degenerate (all x equal).
    // -----------------------------------------------------------------------
    bool weightedOLS(const std::vector<double> &x,
                     const std::vector<double> &y,
                     const std::vector<double> &w,
                     double &out_slope,
                     double &out_intercept) const
    {
        double sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            sw   += w[i];
            swx  += w[i] * x[i];
            swy  += w[i] * y[i];
            swxx += w[i] * x[i] * x[i];
            swxy += w[i] * x[i] * y[i];
        }

        double denom = sw * swxx - swx * swx;
        if (std::fabs(denom) < 1e-14)
        {
            // Degenerate: all x identical — fit a flat line at weighted mean y
            out_slope     = 0.0;
            out_intercept = (sw > 0.0) ? swy / sw : 0.0;
            return false;
        }

        out_slope     = (sw * swxy - swx * swy) / denom;
        out_intercept = (swy - out_slope * swx) / sw;
        return true;
    }

    // -----------------------------------------------------------------------
    // Core IRLS quantile fit.
    // Pinball weight for residual r at quantile tau:
    //   w_i = tau   if r_i > 0   (under-prediction)
    //   w_i = 1-tau if r_i < 0   (over-prediction)
    // Divided by max(|r_i|, epsilon) to form the IRLS re-weighting.
    // -----------------------------------------------------------------------
    void fitQuantile(const std::vector<std::pair<double, double>> &pts)
    {
        size_t n = pts.size();

        std::vector<double> x(n), y(n), w(n, 1.0);
        for (size_t i = 0; i < n; ++i)
        {
            x[i] = pts[i].first;
            y[i] = pts[i].second;
        }

        // Initialise with OLS (tau = 0.5 case) or uniform weights
        weightedOLS(x, y, w, slope, intercept);

        for (int iter = 0; iter < max_iter; ++iter)
        {
            // Compute pinball IRLS weights from current residuals
            for (size_t i = 0; i < n; ++i)
            {
                double r   = y[i] - (slope * x[i] + intercept);
                double abs_r = std::max(std::fabs(r), epsilon);
                // Asymmetric pinball weight
                w[i] = (r >= 0.0 ? tau : (1.0 - tau)) / abs_r;
            }

            double old_slope     = slope;
            double old_intercept = intercept;

            weightedOLS(x, y, w, slope, intercept);

            // Convergence check
            if (std::fabs(slope - old_slope)         < 1e-10 &&
                std::fabs(intercept - old_intercept) < 1e-10)
                break;
        }
    }

public:
    // -----------------------------------------------------------------------
    // Constructor — tau = 0.5 (median regression) by default
    // -----------------------------------------------------------------------
    QuantileRegressionModel()
        : slope(0.0), intercept(0.0), trained(false),
          tau(0.5), max_iter(200), epsilon(1e-6), total(0)
    {}

    // -----------------------------------------------------------------------
    // Hyper-parameter setters / getters
    // -----------------------------------------------------------------------
    void setTau(double t)
    {
        if (t <= 0.0 || t >= 1.0)
            throw std::invalid_argument("tau must be in (0, 1).");
        tau = t;
    }
    double getTau()     const { return tau; }

    void setMaxIter(int m)
    {
        if (m < 1)
            throw std::invalid_argument("max_iter must be >= 1.");
        max_iter = m;
    }
    int getMaxIter()    const { return max_iter; }

    void setEpsilon(double e)
    {
        if (e <= 0.0)
            throw std::invalid_argument("epsilon must be > 0.");
        epsilon = e;
    }
    double getEpsilon() const { return epsilon; }

    // -----------------------------------------------------------------------
    // Fitted coefficient accessors
    // -----------------------------------------------------------------------
    double getSlope()     const { return slope;     }
    double getIntercept() const { return intercept; }

    // -----------------------------------------------------------------------
    // train()  — double keys, matches RegressionModel::train() style.
    // Expects: vector<pair<double,double>> where .first=key, .second=CDF value.
    // Data does NOT need to be pre-sorted.
    // -----------------------------------------------------------------------
    void train(std::vector<std::pair<double, double>> &cdf)
    {
        if (cdf.size() < 2)
            throw std::invalid_argument("Need at least 2 data points to train.");

        std::sort(cdf.begin(), cdf.end(),
                  [](const std::pair<double,double> &a,
                     const std::pair<double,double> &b)
                  { return a.first < b.first; });

        fitQuantile(cdf);
        trained = true;
    }

    // -----------------------------------------------------------------------
    // train_int()  — integer keys, mirrors RegressionModel::train_int().
    // Expects: vector<pair<int,double>> where .first=key, .second=CDF value.
    // -----------------------------------------------------------------------
    void train_int(std::vector<std::pair<int, double>> &cdf)
    {
        if (cdf.size() < 2)
            throw std::invalid_argument("Need at least 2 data points to train.");

        std::vector<std::pair<double, double>> data;
        data.reserve(cdf.size());
        for (const auto &p : cdf)
            data.emplace_back(static_cast<double>(p.first), p.second);

        // train_int receives already-sorted data typically, but sort anyway
        std::sort(data.begin(), data.end(),
                  [](const std::pair<double,double> &a,
                     const std::pair<double,double> &b)
                  { return a.first < b.first; });

        fitQuantile(data);
        trained = true;
    }

    // -----------------------------------------------------------------------
    // predict()  — double key, returns CDF estimate clamped to [0, 1]
    // -----------------------------------------------------------------------
    double predict(double key) const
    {
        if (!trained)
            throw std::runtime_error("Model has not been trained yet.");

        double val = slope * key + intercept;
        return std::max(0.0, std::min(1.0, val));
    }

    // -----------------------------------------------------------------------
    // predict_int()  — integer key, mirrors RegressionModel::predict_int()
    // -----------------------------------------------------------------------
    double predict_int(const int &key) const
    {
        return predict(static_cast<double>(key));
    }

    // -----------------------------------------------------------------------
    // Utility: print fitted line (useful for debugging)
    // -----------------------------------------------------------------------
    void printModel() const
    {
        std::cout << "QuantileRegressionModel (tau=" << tau << "):\n"
                  << "  slope     = " << slope     << "\n"
                  << "  intercept = " << intercept << "\n";
    }

    // -----------------------------------------------------------------------
    // Date helper — mirrors PiecewiseRegressionModel::convertDateStringToTimestamp
    // -----------------------------------------------------------------------
    time_t convertDateStringToTimestamp(const std::string &dateStr)
    {
        std::tm tm = {};
        std::istringstream ss(dateStr);
        ss >> std::get_time(&tm, "%Y-%m-%d");
        if (ss.fail())
            throw std::invalid_argument("Date format is incorrect.");
        return std::mktime(&tm);
    }

    // -----------------------------------------------------------------------
    // Getters / setters for map_cdf_range_k_minwise
    // -----------------------------------------------------------------------
    const std::unordered_map<int, std::set<char>> &getMapCdfRangeKMinwise() const
    {
        return map_cdf_range_k_minwise;
    }
    void setMapCdfRangeKMinwise(const std::unordered_map<int, std::set<char>> &new_map)
    {
        map_cdf_range_k_minwise = new_map;
    }

    // -----------------------------------------------------------------------
    // Getters / setters for map_cdf_range_k_minwise_full
    // -----------------------------------------------------------------------
    const std::unordered_map<int, std::vector<uint16_t>> &getMapCdfRangeKMinwiseFull() const
    {
        return map_cdf_range_k_minwise_full;
    }
    void setMapCdfRangeKMinwiseFull(const std::unordered_map<int, std::vector<uint16_t>> &new_map)
    {
        map_cdf_range_k_minwise_full = new_map;
    }

    // -----------------------------------------------------------------------
    // Getters / setters for total
    // -----------------------------------------------------------------------
    unsigned int getTotal() const        { return total;     }
    void         setTotal(unsigned int t){ total = t;        }

    // -----------------------------------------------------------------------
    // Destructor
    // -----------------------------------------------------------------------
    ~QuantileRegressionModel() = default;
};
