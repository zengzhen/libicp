/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

Authors: Andreas Geiger

libicp is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libicp is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libicp; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#ifndef ICP_POINT_TO_PLANE_H
#define ICP_POINT_TO_PLANE_H

#include "icp.h"
#include <Eigen/Eigen>
#include <Eigen/Geometry>

class IcpPointToPlane : public Icp {

public:
  
  IcpPointToPlane (double *M,const int32_t M_num,const int32_t dim,const int32_t num_neighbors=10,const double flatness=5.0) : Icp(M,M_num,dim) {
      numIters = 0;
      use_normal = true;
      use_projective = false;
    M_normal = computeNormals(num_neighbors,flatness);
  }

  virtual ~IcpPointToPlane () {
    free(M_normal);
  }
  
  Eigen::Matrix4f get_error_jacobian() { return _error_jacobian;};
  
  bool use_normal;
  bool use_projective;

private:
    int numIters;
    
  double fitStep (double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active);
  double fitStepGaussNewton (double *T,const int32_t T_num, Eigen::VectorXf& x, const std::vector<int32_t> &active);
  std::vector<int32_t> getInliers (double *T,const int32_t T_num,const Matrix &R,const Matrix &t,const double indist);
  
  // utility functions to compute normals from the model tree
  void computeNormal (const kdtree::KDTreeResultVector &neighbors,double *M_normal,const double flatness);
  double* computeNormals (const int32_t num_neighbors,const double flatness);
  
  // normals of model points
  double *M_normal;
  
  Eigen::Matrix4f _error_jacobian;
};

#endif // ICP_POINT_TO_PLANE_H
