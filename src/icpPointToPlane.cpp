/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

Authors: Andreas Geiger

openMP support by Manolis Lourakis, Foundation for Research & Technology - Hellas, Heraklion, Greece

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

//#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "icpPointToPlane.h"

using namespace std;

// solve the least squares problem min_x |A*x-b| via SVD
static inline Matrix lssolvesvd(Matrix &A, Matrix &b)
{
  int i;
  Matrix U, S, V;

  A.svd(U, S, V);
  Matrix Uc=U.getMat(0, 0, A.m-1, A.n-1); // compact U
  Matrix Uctb=~Uc*b;
  for(i=0; i<Uctb.m; ++i)
    Uctb.val[i][0]*=(fabs(S.val[i][0])>1E-10)? 1.0/S.val[i][0] : 0.0;

  return V*Uctb;
}

double IcpPointToPlane::fitStepGaussNewton (double *T,const int32_t T_num, Eigen::VectorXf& x, const std::vector<int32_t> &active) {
    
    int i;
    int nact = (int)active.size();
    
    // dimensionality 3
    if (dim==3)
    {        
        // rotation matrix
        Eigen::Matrix3f rx = Rx(x(3));
        Eigen::Matrix3f ry = Ry(x(4));
        Eigen::Matrix3f rz = Rz(x(5));
        
        // derivatives of matrix at euler_angles
        Eigen::Matrix3f rx_p = Rx_prime(x(3));
        Eigen::Matrix3f ry_p = Ry_prime(x(4));
        Eigen::Matrix3f rz_p = Rz_prime(x(5));
        Eigen::Vector3f txyz = x.block<3,1>(0,0);
        
        // init H and b
        Eigen::MatrixXf H(6,6);
        H.setZero();
        Eigen::VectorXf b(6,1);
        b.setZero();
        double chi = 0;
        
        // establish correspondences
//         #pragma omp parallel for private(i) default(none) shared(T,active,nact,H,b,txyz,rx,ry,rz,rx_p,ry_p,rz_p,chi) // schedule (dynamic,2)
        
        for (i=0; i<nact; i++) {
            // kd tree query + result
            std::vector<float>         query(dim);
            kdtree::KDTreeResultVector result;
            
            // get index of active point
            int32_t idx = active[i];
            
            // transform point according to R|t
            Eigen::Vector3f p(T[idx*3+0], T[idx*3+1], T[idx*3+2]);
            Eigen::Vector3f z_hat = ((rx*ry)*rz)*p + txyz;
            
            query[0] = z_hat[0];
            query[1] = z_hat[1];
            query[2] = z_hat[2];
            
            if(use_projective)
            {
                // projective data association
                Eigen::Vector3f projected_z_hat = CamMatrix*z_hat;
                projected_z_hat[0] = projected_z_hat[0]/projected_z_hat[2];
                projected_z_hat[1] = projected_z_hat[1]/projected_z_hat[2];
                projected_z_hat[2] = 1;
                int coord_r = round(projected_z_hat[0]);
                int coord_c = round(projected_z_hat[1]);
                int32_t corres_index = coord_r*image_w + coord_c;
                
                if(coord_r>=0 & coord_r<image_h & coord_c>=0 & coord_c<image_w)
                {
                    double dx = M_data[corres_index][0];
                    double dy = M_data[corres_index][1];
                    double dz = M_data[corres_index][2];
                    
                    double nx = M_normal[corres_index*3+0];
                    double ny = M_normal[corres_index*3+1];
                    double nz = M_normal[corres_index*3+2];
                    
                    // error
                    Eigen::Vector3f z(dx, dy, dz);
                    Eigen::Vector3f e = z_hat - z;
                    
                    if(e.transpose()*e < 0.5)
                    {
                        // setup least squares system
                        Eigen::MatrixXf J(3,6);
                        J.block<3,3>(0,0) = Eigen::Matrix3f::Identity(3,3);
                        J.block<3,1>(0,3) = ((rx_p*ry)*rz)*p;
                        J.block<3,1>(0,4) = ((rx*ry_p)*rz)*p;
                        J.block<3,1>(0,5) = ((rx*ry)*rz_p)*p;
                        
                        H = H + J.transpose()*J;
                        b = b + J.transpose()*e;
                        chi = chi + e.transpose()*e;
                    }
                }
            }else{      
                // search nearest neighbor
                M_tree->n_nearest(query,1,result);
                assert(result.size()!=0); // check if NN search failed
                
                // model point
                double dx = M_tree->the_data[result[0].idx][0];
                double dy = M_tree->the_data[result[0].idx][1];
                double dz = M_tree->the_data[result[0].idx][2];
                
//                 if(sqrt((query[0]-dx)*(query[0]-dx)+(query[1]-dy)*(query[1]-dy)+(query[2]-dz)*(query[2]-dz)) > 10)
//                 {
//                     printf("input point: %f %f %f\n",  query[0], query[1], query[2]);
//                     printf("model point: %f %f %f\n",  dx, dy, dz);
//                     continue;
//                 }
                
                // model point normal
                double nx = M_normal[result[0].idx*3+0];
                double ny = M_normal[result[0].idx*3+1];
                double nz = M_normal[result[0].idx*3+2];
                Eigen::Vector3f normal_vector(nx, ny, nz);
                
                // error
                if(use_normal)
                {
                    Eigen::Vector3f z(dx, dy, dz);
                    float e = normal_vector.transpose()*(z_hat - z);
                    
                    // setup least squares system
                    Eigen::MatrixXf J_woN(3,6);
                    J_woN.block<3,3>(0,0) = Eigen::Matrix3f::Identity(3,3);
                    J_woN.block<3,1>(0,3) = ((rx_p*ry)*rz)*p;
                    J_woN.block<3,1>(0,4) = ((rx*ry_p)*rz)*p;
                    J_woN.block<3,1>(0,5) = ((rx*ry)*rz_p)*p;
                    Eigen::MatrixXf J(1,6);
                    J = normal_vector.transpose()*J_woN;
                    
                    H = H + J.transpose()*J;
                    b = b + J.transpose()*e;
                    chi = chi + e*e;
                }else{
                    Eigen::Vector3f z(dx, dy, dz);
                    Eigen::Vector3f e = z_hat - z;
                    
                    // setup least squares system
                    Eigen::MatrixXf J(3,6);
                    J.block<3,3>(0,0) = Eigen::Matrix3f::Identity(3,3);
                    J.block<3,1>(0,3) = ((rx_p*ry)*rz)*p;
                    J.block<3,1>(0,4) = ((rx*ry_p)*rz)*p;
                    J.block<3,1>(0,5) = ((rx*ry)*rz_p)*p;
                    
                    H = H + J.transpose()*J;
                    b = b + J.transpose()*e;
                    chi = chi + e.transpose()*e;
                }
            }
        }
        
        hessian = H.cast<double>();
        
        Matrix A(6,6);
        Matrix bb(6,1);
        for (int32_t k=0; k<H.rows(); k++)
            for (int32_t j=0; j<H.cols(); j++)
                A.val[k][j] = H(k,j);
           
        for (int32_t k=0; k<b.rows(); k++)
            bb.val[k][0] = -b[k];
        
        // solve linear least squares
        #if 1
            // use the normal equations
            Matrix A_ = A;
            Matrix bb_ = bb;
        
            if (!bb_.solve(A_)) return -1; // failure
        #else
            // use SVD which is slower but more stable numerically
            Matrix bb_=lssolvesvd(A, bb);
        #endif
        
        Eigen::VectorXf dx(6,1);
        dx << bb_.val[0][0], bb_.val[1][0], bb_.val[2][0], bb_.val[3][0], bb_.val[4][0], bb_.val[5][0];
        x = x + dx;
        
        Eigen::Matrix3f dR = Rx(dx(3))*Ry(dx(4))*Rz(dx(5));
        Matrix varR_(3,3);
        for (int32_t k=0; k<dR.rows(); k++)
            for (int32_t j=0; j<dR.cols(); j++)
                varR_.val[k][j] = dR(k,j);
        Matrix vart_(3,1);
        for (int32_t k=0; k<3; k++)
            vart_.val[k][0] = dx[k];
        
//         std::cout << "dx = " << dx.transpose() << std::endl;
//         std::cout << "x = " << x.transpose() << std::endl;
//         std::cout << "chi = " << chi << std::endl;
        
        double step = max((varR_-Matrix::eye(3)).l2norm(),vart_.l2norm());
        return step;
    }else{
        printf("only works with 3D points\n");
        exit(1);
    }
    
    // failure
    return 0;
}


// Also see (3d part): "Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration" (Kok-Lim Low)
double IcpPointToPlane::fitStep (double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active) {

  int i;
  int nact = (int)active.size();

  // init matrix for point correspondences
  Matrix p_m(nact,dim); // model
  Matrix p_t(nact,dim); // template
  
  // dimensionality 2
  if (dim==2) {

    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1];
    double r10 = R.val[1][0]; double r11 = R.val[1][1];
    double t0  = t.val[0][0]; double t1  = t.val[1][0];

    // init A and b
    Matrix A(nact,3);
    Matrix b(nact,1);

    // establish correspondences
#pragma omp parallel for private(i) default(none) shared(T,active,nact,p_m,p_t,A,b,r00,r01,r10,r11,t0,t1) // schedule (dynamic,2)
    for (i=0; i<nact; i++) {
      // kd tree query + result
      std::vector<float>         query(dim);
      kdtree::KDTreeResultVector result;

      // get index of active point
      int32_t idx = active[i];

      // transform point according to R|t
      query[0] = (float)(r00*T[idx*2+0] + r01*T[idx*2+1] + t0);
      query[1] = (float)(r10*T[idx*2+0] + r11*T[idx*2+1] + t1);

      // search nearest neighbor
      M_tree->n_nearest(query,1,result);

      // model point
      double dx = M_tree->the_data[result[0].idx][0];
      double dy = M_tree->the_data[result[0].idx][1];

      // model point normal
      double nx = M_normal[result[0].idx*2+0];
      double ny = M_normal[result[0].idx*2+1];

      // template point
      double sx = query[0];
      double sy = query[1];

      // setup least squares system
      A.val[i][0] = ny*sx-nx*sy;
      A.val[i][1] = nx;
      A.val[i][2] = ny;
      b.val[i][0] = nx*(dx-sx) + ny*(dy-sy); //nx*dx+ny*dy-nx*sx-ny*sy;    
    }

    // solve linear least squares
#if 1
    // use the normal equations
    Matrix A_ = ~A*A;
    Matrix b_ = ~A*b;

    if (!b_.solve(A_)) return 0; // failure
#else
    // use SVD which is slower but more stable numerically
    Matrix b_=lssolvesvd(A, b);
#endif

    // rotation matrix
    Matrix R_ = Matrix::eye(2);
    R_.val[0][1] = -b_.val[0][0];
    R_.val[1][0] = +b_.val[0][0];

    // orthonormalized rotation matrix
    Matrix U,W,V;
    R_.svd(U,W,V);
    R_ = U*~V;  

    // fix improper matrix problem
    if (R_.det()<0){
      Matrix B = Matrix::eye(dim);
      B.val[dim-1][dim-1] = R_.det();
      R_ = V*B*~U;
    }

    // translation vector
    Matrix t_(2,1);
    t_.val[0][0] = b_.val[1][0];
    t_.val[1][0] = b_.val[2][0];

    // compose: R|t = R_|t_ * R|t
    R = R_*R;
    t = R_*t+t_;
    return max((R_-Matrix::eye(2)).l2norm(),t_.l2norm());
   
  // dimensionality 3
  } else {
    
    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1]; double r02 = R.val[0][2];
    double r10 = R.val[1][0]; double r11 = R.val[1][1]; double r12 = R.val[1][2];
    double r20 = R.val[2][0]; double r21 = R.val[2][1]; double r22 = R.val[2][2];
    double t0  = t.val[0][0]; double t1  = t.val[1][0]; double t2  = t.val[2][0];

    // init A and b
    Matrix A(nact,6);
    Matrix b(nact,1);

    // establish correspondences
#pragma omp parallel for private(i) default(none) shared(T,active,nact,p_m,p_t,A,b,r00,r01,r02,r10,r11,r12,r20,r21,r22,t0,t1,t2) // schedule (dynamic,2)
    
    for (i=0; i<nact; i++) {
      // kd tree query + result
      std::vector<float>         query(dim);
      kdtree::KDTreeResultVector result;

      // get index of active point
      int32_t idx = active[i];

      // transform point according to R|t
      query[0] = (float)(r00*T[idx*3+0] + r01*T[idx*3+1] + r02*T[idx*3+2] + t0);
      query[1] = (float)(r10*T[idx*3+0] + r11*T[idx*3+1] + r12*T[idx*3+2] + t1);
      query[2] = (float)(r20*T[idx*3+0] + r21*T[idx*3+1] + r22*T[idx*3+2] + t2);

      // search nearest neighbor
      M_tree->n_nearest(query,1,result);
      //assert(result.size()!=0); // check if NN search failed

      // model point
      double dx = M_tree->the_data[result[0].idx][0];
      double dy = M_tree->the_data[result[0].idx][1];
      double dz = M_tree->the_data[result[0].idx][2];

      // model point normal
      double nx = M_normal[result[0].idx*3+0];
      double ny = M_normal[result[0].idx*3+1];
      double nz = M_normal[result[0].idx*3+2];

      // template point
      double sx = query[0];
      double sy = query[1];
      double sz = query[2];

      // setup least squares system
      A.val[i][0] = nz*sy-ny*sz;
      A.val[i][1] = nx*sz-nz*sx;
      A.val[i][2] = ny*sx-nx*sy;
      A.val[i][3] = nx;
      A.val[i][4] = ny;
      A.val[i][5] = nz;
      b.val[i][0] = nx*(dx-sx) + ny*(dy-sy) + nz*(dz-sz); //nx*dx+ny*dy+nz*dz-nx*sx-ny*sy-nz*sz;    
    }

    // solve linear least squares
#if 1
    // use the normal equations
    Matrix A_ = ~A*A;
    Matrix b_ = ~A*b;

    if (!b_.solve(A_)) return -1; // failure
#else
    // use SVD which is slower but more stable numerically
    Matrix b_=lssolvesvd(A, b);
#endif

    // rotation matrix
    Matrix R_ = Matrix::eye(3);
    R_.val[0][1] = -b_.val[2][0];
    R_.val[1][0] = +b_.val[2][0];
    R_.val[0][2] = +b_.val[1][0];
    R_.val[2][0] = -b_.val[1][0];
    R_.val[1][2] = -b_.val[0][0];
    R_.val[2][1] = +b_.val[0][0];

    // orthonormalized rotation matrix
    Matrix U,W,V;
    R_.svd(U,W,V);
    R_ = U*~V;  

    // fix improper matrix problem
    if (R_.det()<0){
      Matrix B = Matrix::eye(dim);
      B.val[dim-1][dim-1] = R_.det();
      R_ = V*B*~U;
    }

    // translation vector
    Matrix t_(3,1);
    t_.val[0][0] = b_.val[3][0];
    t_.val[1][0] = b_.val[4][0];
    t_.val[2][0] = b_.val[5][0];

    // compose: R|t = R_|t_ * R|t
    R = R_*R;
    t = R_*t+t_;
    
    // calculate the Jacobian of the alignment error
//     _error_jacobian.setIdentity();
//     r00 = R.val[0][0]; r01 = R.val[0][1]; r02 = R.val[0][2];
//     r10 = R.val[1][0]; r11 = R.val[1][1]; r12 = R.val[1][2];
//     r20 = R.val[2][0]; r21 = R.val[2][1]; r22 = R.val[2][2];
//     t0  = t.val[0][0]; t1  = t.val[1][0]; t2  = t.val[2][0];
//     
//     double alpha, beta, gamma;
//     alpha = std::atan2(r21, r22);
//     beta = std::asin(-r20);
//     gamma = std::acos(r00/std::cos(beta));
//     
//     double d_alpha = 0;
//     double d_beta = 0;
//     double d_gamma = 0;
//     double d_tx = 0;
//     double d_ty = 0;
//     double d_tz = 0;
//     
//     for (i=0; i<nact; i++) 
//     {
//         std::vector<float>         query(dim);
//         kdtree::KDTreeResultVector result;
//         
//         // get index of active point
//         int32_t idx = active[i];
//         
//         // transform point according to R|t
//         query[0] = (float)(r00*T[idx*3+0] + r01*T[idx*3+1] + r02*T[idx*3+2] + t0);
//         query[1] = (float)(r10*T[idx*3+0] + r11*T[idx*3+1] + r12*T[idx*3+2] + t1);
//         query[2] = (float)(r20*T[idx*3+0] + r21*T[idx*3+1] + r22*T[idx*3+2] + t2);
//         
//         // search nearest neighbor
//         M_tree->n_nearest(query,1,result);
//         //assert(result.size()!=0); // check if NN search failed
//         
//         // model point
//         double dx = M_tree->the_data[result[0].idx][0];
//         double dy = M_tree->the_data[result[0].idx][1];
//         double dz = M_tree->the_data[result[0].idx][2];
//         
//         // model point normal
//         double nx = M_normal[result[0].idx*3+0];
//         double ny = M_normal[result[0].idx*3+1];
//         double nz = M_normal[result[0].idx*3+2];
//         
//         // original point
//         double ox = T[idx*3+0];
//         double oy = T[idx*3+1];
//         double oz = T[idx*3+2];
//         
//         // template point
//         double sx = query[0];
//         double sy = query[1];
//         double sz = query[2];
//         
//         // point difference
//         double dpx = sx - dx;
//         double dpy = sy - dy;
//         double dpz = sz - dz;
//         
//         // derivative: dM(R,t)/d(alpha, beta, gamma, tx, ty, tz)
//         // file:///home/zengzhen/Downloads/Linear_Least-Squares_Optimization_for_Point-to-Pla.pdf
//         double temp=0;
//         temp += 2*dpx*((sin(beta)*sin(alpha)+cos(gamma)*sin(beta)*cos(alpha))*oy
//                       +(sin(gamma)*cos(alpha)-cos(gamma)*sin(beta)*sin(alpha))*oz)*nx*nx;
//         temp += 2*dpy*((-cos(beta)*sin(alpha)+sin(gamma)*sin(beta)*cos(alpha))*oy
//                       +(-cos(gamma)*cos(alpha)-sin(gamma)*sin(beta)*sin(alpha))*oz)*ny*ny;
//         temp += 2*dpz*(cos(beta)*cos(alpha)*oy-cos(beta)*sin(alpha)*oz)*nz*nz;
//         d_alpha += temp;
//         
//         temp = 0;
//         temp += 2*dpx*(-cos(gamma)*sin(beta)*ox + cos(gamma)*cos(beta)*sin(alpha)*oy
//                       +cos(gamma)*cos(beta)*cos(alpha)*oz)*nx*nx;
//         temp += 2*dpy*(-sin(gamma)*sin(beta)*ox + sin(gamma)*cos(beta)*sin(alpha)*oy
//                       +sin(gamma)*cos(beta)*cos(alpha)*oz)*ny*ny;
//         temp += 2*dpz*(-cos(beta)*ox - sin(beta)*sin(alpha)*oy
//                       -sin(beta)*cos(alpha)*oz)*nz*nz;
//         d_beta += temp;
//         
//         temp = 0;
//         temp += 2*dpx*(-sin(gamma)*cos(beta)*ox + (-cos(gamma)*cos(alpha)-sin(gamma)*sin(beta)*sin(alpha))*oy
//                       +(cos(gamma)*sin(alpha)-sin(gamma)*sin(beta)*cos(alpha))*oz)*nx*nx;
//         temp += 2*dpy*(-cos(gamma)*cos(beta)*ox + (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*oy
//                       +(sin(gamma)*sin(alpha)+cos(gamma)*sin(beta)*cos(alpha))*oz)*ny*ny;
//         d_gamma += temp;
//         
//         d_tx += 2*dpx*nx*nx;
//         d_ty += 2*dpy*ny*ny;
//         d_tz += 2*dpz*nz*nz;
//     }
    
    
    
    
    return max((R_-Matrix::eye(3)).l2norm(),t_.l2norm());
  }
  
  // failure
  return 0;
}

std::vector<int32_t> IcpPointToPlane::getInliers (double *T,const int32_t T_num,const Matrix &R,const Matrix &t,const double indist) {
  
   // init inlier vector + query point + query result
  vector<int32_t>            inliers;
  std::vector<float>         query(dim);
  kdtree::KDTreeResultVector neighbor;
  
  // dimensionality 2
  if (dim==2) {
  
    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1];
    double r10 = R.val[1][0]; double r11 = R.val[1][1];
    double t0  = t.val[0][0]; double t1  = t.val[1][0];

    // check for all points if they are inliers
    for (int32_t i=0; i<T_num; i++) {

      // transform point according to R|t
      double sx = r00*T[i*2+0] + r01*T[i*2+1] + t0; query[0] = (float)sx;
      double sy = r10*T[i*2+0] + r11*T[i*2+1] + t1; query[1] = (float)sy;

      // search nearest neighbor
      M_tree->n_nearest(query,1,neighbor);
      //assert(result.size()!=0); // check if NN search failed

      // model point
      double dx = M_tree->the_data[neighbor[0].idx][0];
      double dy = M_tree->the_data[neighbor[0].idx][1];

      // model point normal
      double nx = M_normal[neighbor[0].idx*2+0];
      double ny = M_normal[neighbor[0].idx*2+1];

      // check if it is an inlier
      if ((sx-dx)*nx+(sy-dy)*ny<indist)
        inliers.push_back(i);
    }
    
  // dimensionality 3
  } else {
    
    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1]; double r02 = R.val[0][2];
    double r10 = R.val[1][0]; double r11 = R.val[1][1]; double r12 = R.val[1][2];
    double r20 = R.val[2][0]; double r21 = R.val[2][1]; double r22 = R.val[2][2];
    double t0  = t.val[0][0]; double t1  = t.val[1][0]; double t2  = t.val[2][0];

    // check for all points if they are inliers
    for (int32_t i=0; i<T_num; i++) {

      // transform point according to R|t
      double sx = r00*T[i*3+0] + r01*T[i*3+1] + r02*T[i*3+2] + t0; query[0] = (float)sx;
      double sy = r10*T[i*3+0] + r11*T[i*3+1] + r12*T[i*3+2] + t1; query[1] = (float)sy;
      double sz = r20*T[i*3+0] + r21*T[i*3+1] + r22*T[i*3+2] + t2; query[2] = (float)sz;

      // search nearest neighbor
      M_tree->n_nearest(query,1,neighbor);

      // model point
      double dx = M_tree->the_data[neighbor[0].idx][0];
      double dy = M_tree->the_data[neighbor[0].idx][1];
      double dz = M_tree->the_data[neighbor[0].idx][2];

      // model point normal
      double nx = M_normal[neighbor[0].idx*3+0];
      double ny = M_normal[neighbor[0].idx*3+1];
      double nz = M_normal[neighbor[0].idx*3+2];

      // check if it is an inlier
//       if(sqrt((sx-dx)*(sx-dx)+(sy-dy)*(sy-dy)+(sz-dz)*(sz-dz)) < indist)
//       if ((sx-dx)*nx+(sy-dy)*ny+(sz-dz)*nz<indist)
      if (fabs((sx-dx)*nx)+fabs((sy-dy)*ny)+fabs((sz-dz)*nz)<indist)
        inliers.push_back(i);
    }
  }
  
  // return vector with inliers
  return inliers;
}

void IcpPointToPlane::computeNormal (const kdtree::KDTreeResultVector &neighbors,double *M_normal,const double flatness) {
  
  // dimensionality 2
  if (dim==2) {
    
    // extract neighbors
    Matrix P(neighbors.size(),2);
    Matrix mu(1,2);
    for (uint32_t i=0; i<neighbors.size(); i++) {
      double x = M_tree->the_data[neighbors[i].idx][0];
      double y = M_tree->the_data[neighbors[i].idx][1];
      P.val[i][0] = x;
      P.val[i][1] = y;
      mu.val[0][0] += x;
      mu.val[0][1] += y;
    }

    // zero mean
    mu       = mu/(double)neighbors.size();
    Matrix Q = P - Matrix::ones(neighbors.size(),1)*mu;

    // principal component analysis
    Matrix H = ~Q*Q;
    Matrix U,W,V;
    H.svd(U,W,V);

    // normal
    M_normal[0] = U.val[0][1];
    M_normal[1] = U.val[1][1];
  
  // dimensionality 3
  } else {
    
    // extract neighbors
    Matrix P(neighbors.size(),3);
    Matrix mu(1,3);
    for (uint32_t i=0; i<neighbors.size(); i++) {
      double x = M_tree->the_data[neighbors[i].idx][0];
      double y = M_tree->the_data[neighbors[i].idx][1];
      double z = M_tree->the_data[neighbors[i].idx][2];
      P.val[i][0] = x;
      P.val[i][1] = y;
      P.val[i][2] = z;
      mu.val[0][0] += x;
      mu.val[0][1] += y;
      mu.val[0][2] += z;
    }

    // zero mean
    mu       = mu/(double)neighbors.size();
    Matrix Q = P - Matrix::ones(neighbors.size(),1)*mu;

    // principal component analysis
    Matrix H = ~Q*Q;
    Matrix U,W,V;
    H.svd(U,W,V);

    // normal
    M_normal[0] = U.val[0][2];
    M_normal[1] = U.val[1][2];
    M_normal[2] = U.val[2][2];
  }
}

double* IcpPointToPlane::computeNormals (const int32_t num_neighbors,const double flatness) {
  double *M_normal = (double*)malloc(M_tree->N*dim*sizeof(double));
  kdtree::KDTreeResultVector neighbors;
  for (int32_t i=0; i<M_tree->N; i++) {
    M_tree->n_nearest_around_point(i,0,num_neighbors,neighbors);
    if (dim==2) computeNormal(neighbors,M_normal+i*2,flatness);
    else        computeNormal(neighbors,M_normal+i*3,flatness);
  }
  return M_normal;
}
