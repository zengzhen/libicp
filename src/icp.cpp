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

#include "icp.h"

using namespace std;

Icp::Icp (double *M,const int32_t M_num,const int32_t dim) :
  dim(dim), max_iter(200), min_delta(1e-2) {
      
  // check for correct dimensionality
  if (dim!=2 && dim!=3) {
    cout << "ERROR: LIBICP works only for data of dimensionality 2 or 3" << endl;
    M_tree = 0;
    return;
  }
  
  // check for minimum number of points
  if (M_num<5) {
    cout << "ERROR: LIBICP works only with at least 5 model points" << endl;
    M_tree = 0;
    return;
  }
 
  // copy model points to M_data
  M_data.resize(boost::extents[M_num][dim]);
  for (int32_t m=0; m<M_num; m++)
    for (int32_t n=0; n<dim; n++)
      M_data[m][n] = (float)M[m*dim+n];

  // build a kd tree from the model point cloud
  M_tree = new kdtree::KDTree(M_data);
  CamMatrix.setIdentity();
}

Icp::~Icp () {
  if (M_tree)
    delete M_tree;
}

void Icp::setCamParms(double fx, double fy, double ox, double oy)
{
    CamMatrix(0,0) = fx;
    CamMatrix(1,1) = fy;
    CamMatrix(0,2) = ox;
    CamMatrix(1,2) = oy;
}

void Icp::setImageSize(int w, int h)
{
    image_w = w;
    image_h = h;
}

Eigen::Vector3f Icp::poseToEuler(Eigen::Matrix4f pose)
{
    Eigen::Matrix3f rotation = pose.block<3,3>(0,0);
    Eigen::Vector3f euler = rotation.eulerAngles(0,1,2);
    return euler;
}

Eigen::Vector3f Icp::poseToTrans(Eigen::Matrix4f pose)
{
    Eigen::Vector3f trans = pose.block<3,1>(0,3);
    return trans;
}

Eigen::Matrix3f Icp::Rx(double angle)
{
    Eigen::Matrix3f R;    
    double c=cos(angle);
    double s=sin(angle);
    R << 1, 0, 0, 0, c, -s, 0, s, c;
    
    return R;
}

Eigen::Matrix3f Icp::Ry(double angle)
{
    Eigen::Matrix3f R;    
    double c=cos(angle);
    double s=sin(angle);
    R << c, 0, s, 0, 1, 0, -s, 0, c;
    
    return R;
}

Eigen::Matrix3f Icp::Rz(double angle)
{
    Eigen::Matrix3f R;    
    double c=cos(angle);
    double s=sin(angle);
    R << c, -s, 0, s, c, 0, 0, 0, 1;
    
    return R;
}

Eigen::Matrix3f Icp::Rx_prime(double angle)
{
    Eigen::Matrix3f R;    
    double c=cos(angle);
    double s=sin(angle);
    R << 0, 0, 0, 0, -s, -c, 0, c, -s;
    
    return R;
}

Eigen::Matrix3f Icp::Ry_prime(double angle)
{
    Eigen::Matrix3f R;    
    double c=cos(angle);
    double s=sin(angle);
    R << -s, 0, c, 0, 0, 0, -c, 0, -s;
    
    return R;
    
}

Eigen::Matrix3f Icp::Rz_prime(double angle)
{
    Eigen::Matrix3f R;    
    double c=cos(angle);
    double s=sin(angle);
    R << -s, -c, 0, c, -s, 0, 0, 0, 0;
    
    return R;
}

bool Icp::fit (double *T,const int32_t T_num,Matrix &R,Matrix &t,const double indist) {
  
  // make sure we have a model tree
  if (!M_tree) {
    cout << "ERROR: No model available." << endl;
    return false;
  }
  
  // check for minimum number of points
  if (T_num<5) {
    cout << "ERROR: Icp works only with at least 5 template points" << endl;
    return false;
  }
  
  // set active points
  vector<int32_t> active;
  if (indist<=0) {
    active.clear();
    for (int32_t i=0; i<T_num; i++)
      active.push_back(i);
  } else {
    active = getInliers(T,T_num,R,t,indist);
  }
  
  // run icp
  bool converged = fitIterate(T,T_num,R,t,active);
  return converged;
}

Eigen::Matrix<double, 6, 6> Icp::getHessian()
{
    Eigen::Matrix<double, 6, 6> information(hessian);
    return information;
}

bool Icp::fitGaussNewton (double *T,const int32_t T_num,Eigen::VectorXf& x,const double indist) {
    
    // make sure we have a model tree
    if (!M_tree) {
        cout << "ERROR: No model available." << endl;
        return false;
    }
    
    // check for minimum number of points
    if (T_num<5) {
        cout << "ERROR: Icp works only with at least 5 template points" << endl;
        return false;
    }
    
    // get rotation matrix and  translation vector 
    Eigen::Matrix3f RR = Rx(x[3])*Ry(x[4])*Rz(x[5]);
    Matrix R(3,3);
    for (int32_t i=0; i<RR.rows(); i++)
        for (int32_t j=0; j<RR.cols(); j++)
            R.val[i][j] = RR(i,j);
    Matrix t(3,1);
    for (int32_t i=0; i<3; i++)
        t.val[i][0] = x(i);
    
    cout << "initial guess:\n";
    cout << "R:" << endl << R << endl << endl;
    cout << "t:" << endl << ~t << endl << endl;
    
    // set active points
    vector<int32_t> active;
    if (indist<=0) {
        active.clear();
        for (int32_t i=0; i<T_num; i++)
            active.push_back(i);
    } else {
        active = getInliers(T,T_num,R,t,indist);
        std::cout << "# inlieres = " << (int)active.size() << std::endl;
    }
    
    // run icp
    bool converged = fitIterateGaussNewton(T,T_num,x,active);
    return converged;
}

bool Icp::fitIterate(double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active) {
  
  // check if we have at least 5 active points
  if (active.size()<5)
    return false;
  
  // iterate until convergence
  int32_t iter=0;
  for (iter=0; iter<max_iter; iter++)
    if (fitStep(T,T_num,R,t,active)<min_delta)
      break;
    
  if(iter==max_iter)
      return false;
  else
      return true;
}

bool Icp::fitIterateGaussNewton(double *T,const int32_t T_num,Eigen::VectorXf& x,const std::vector<int32_t> &active) {
    
    // check if we have at least 5 active points
    if (active.size()<5)
        return false;
    
    // iterate until convergence
    int32_t iter=0;
    for (iter=0; iter<max_iter; iter++)
    {
        std::cout << "iter #" << iter << ": ";
        double delta = fitStepGaussNewton(T,T_num,x,active);
        
        if (delta==-1)
        {
            printf("cannot solve H*dx=b\n");
            exit(1);
        }
        if (delta<min_delta)
        {
            printf("*****************************************\n");
            printf("converged.\n");
            std::cout << "x: " << x.transpose() << std::endl;
            Eigen::Matrix3f RR = Rx(x[3])*Ry(x[4])*Rz(x[5]);
            std::cout << "result Gauss Newton: \n R: \n" << RR << std::endl;
            std::cout << "t: \n" << x.block<3,1>(0,0).transpose() << std::endl;
            break;
        }
    }
    
    if(iter==max_iter)
    {
        printf("*****************************************\n");
        printf("not converged within max_iter.\n");
        return false;
    }else
        return true;
}
