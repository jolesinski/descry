#ifndef DESCRY_LOGGER_H
#define DESCRY_LOGGER_H

#include <spdlog/spdlog.h>

namespace descry { namespace logger {

using handle = std::shared_ptr<spdlog::logger>;

static constexpr auto LOGGER_NAME = "descry";

// TODO: configure
inline void init() {
    spdlog::set_async_mode(8192);
    spdlog::stdout_color_mt(LOGGER_NAME);
}

inline auto get() {
    return spdlog::get(LOGGER_NAME);
}

}}

#endif //DESCRY_LOGGER_H
