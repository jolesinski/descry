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

    Latency(bool enabled) : disabled_(!enabled) {
        if (enabled)
            log_ = logger::get();

        if (!log_)
            disabled_ = true;
    }

    void log_duration(const std::string& name, time_point started, time_point finished) {
        if (disabled_) return;
        auto duration = std::chrono::duration_cast<resolution>(finished - started);
        log_->info("{} took: {}ms", name, duration.count());
    }

    template <typename NameType>
    void start(NameType&& name) {
        if (disabled_) return;
        auto& stamp = start_stamps[std::forward<NameType>(name)];
        stamp = clock::now();
    }

    template <typename NameType>
    void finish(NameType&& name) {
        if (disabled_) return;
        auto finished = clock::now();
        auto name_str = std::string(std::forward<NameType>(name));
        log_duration(name, start_stamps[name], finished);
    }

    void finish() {
        if (disabled_) return;
        auto finished = clock::now();
        for (auto& elem : start_stamps)
            log_duration(elem.first, elem.second, finished);
        start_stamps.clear();
    }

    template <typename NameType>
    void restart(NameType&& name) {
        finish();
        start(std::forward<NameType>(name));
    }

private:
    bool disabled_;
    logger::handle log_;
    std::unordered_map<std::string, time_point> start_stamps;
};

template <typename NameType>
Latency measure_latency(NameType&& name, bool enabled = true) {
    auto latency = Latency(enabled);
    latency.start(std::forward<NameType>(name));

    return latency;
}

class ScopedLatency {
public:
    template <typename NameType>
    ScopedLatency(NameType&& name, bool enabled) :
            latency_(measure_latency(std::forward<NameType>(name), enabled)) {}

    ~ScopedLatency() {
        latency_.finish();
    }

private:
    Latency latency_;
};


template <typename NameType>
ScopedLatency measure_scope_latency(NameType&& name, bool enabled = true) {
    return ScopedLatency(std::forward<NameType>(name), enabled);
}

}

#endif //DESCRY_LATENCY_H
