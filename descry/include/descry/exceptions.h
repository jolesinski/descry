#ifndef DESCRY_EXCEPTIONS_H
#define DESCRY_EXCEPTIONS_H

#include <pcl/exceptions.h>

#define DESCRY_THROW(exception, message) PCL_THROW_EXCEPTION(exception, message)
#define DESCRY_DEFINE_EXCEPTION(Exception) \
class Exception : public pcl::PCLException { \
public: \
Exception(const std::string &error_description = "", \
          const char *file_name = NULL, \
          const char *function_name = NULL, \
          unsigned line_number = 0) \
        : pcl::PCLException(error_description, file_name, function_name, line_number) {} \
};

namespace descry {

DESCRY_DEFINE_EXCEPTION(NotConfiguredException)
DESCRY_DEFINE_EXCEPTION(NoMemoryToTransferException)

}

#endif //DESCRY_EXCEPTIONS_H
