#ifndef DESCRY_LATENCY_H
#define DESCRY_LATENCY_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <descry/logger.h>

namespace descry {

// Utility for time measurements logging
class Latency {
public:
    using clock = std::chrono::steady_clock;
    using time_point = std::chrono::time_point<clock>;
    using resolution = std::chrono::milliseconds;

    Latency() : log_(logger::get()) {}

    void log_duration(const std::string& name, time_point started, time_point finished) {
        auto duration = std::chrono::duration_cast<resolution>(finished - started);
        log_->info("{} took: {}ms", name, duration.count());
    }

    template <typename NameType>
    void start(NameType&& name) {
        auto& stamp = start_stamps[std::forward<NameType>(name)];
        stamp = clock::now();
    }

    template <typename NameType>
    void finish(NameType&& name) {
        auto finished = clock::now();
        auto name_str = std::string(std::forward<NameType>(name));
        log_duration(name, start_stamps[name], finished);
    }

    void finish() {
        auto finished = clock::now();
        for (auto& elem : start_stamps)
            log_duration(elem.first, elem.second, finished);
        start_stamps.clear();
    }

private:
    logger::handle log_;
    std::unordered_map<std::string, time_point> start_stamps;
};

template <typename NameType>
Latency measure_latency(NameType&& name) {
    auto latency = Latency();
    latency.start(std::forward<NameType>(name));

    return latency;
}

}

#endif //DESCRY_LATENCY_H
