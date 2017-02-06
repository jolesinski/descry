
#ifndef DESCRY_CUPCL_LRF_H
#define DESCRY_CUPCL_LRF_H

namespace descry { namespace cupcl {

// pcl::ReferenceFrame is not CUDA friendly
struct LRF
{
  union
  {
    float rf[9];
    struct
    {
      float x_axis[3];
      float y_axis[3];
      float z_axis[3];
    };
  };
};

} }

#endif //DESCRY_CUPCL_LRF_H
