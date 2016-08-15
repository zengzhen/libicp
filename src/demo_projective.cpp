/*
 * Copyright 2011. All rights reserved.
 * Institute of Measurement and Control Systems
 * Karlsruhe Institute of Technology, Germany
 * 
 * This file is part of libicp.
 * Authors: Andreas Geiger
 * 
 * libicp is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 3 of the License, or any later version.
 * 
 * libicp is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * libicp; if not, write to the Free Software Foundation, Inc., 51 Franklin
 * Street, Fifth Floor, Boston, MA 02110-1301, USA 
 */

// Demo program showing how libicp can be used

#include <iostream>
#include "icpPointToPlane.h"

using namespace std;

int main (int argc, char** argv) {
    
    // define a 3 dim problem with 10000 model points
    // and 10000 template points:
    int32_t dim = 3;
    
    int image_w = 100;
    int image_h = 100;
    int32_t num = 10000;
    double fx=0.525, fy=0.525, ox=50, oy=50; 
    
    // allocate model and template memory
    double* M = (double*)calloc(3*num,sizeof(double));
    double* T = (double*)calloc(3*num,sizeof(double));
    
    // set model and template points
    Eigen::Vector3f eulers(0.1, 0.3, 0);
    Eigen::Matrix3f RR = IcpPointToPlane::Rx(eulers[0])*IcpPointToPlane::Ry(eulers[1])*IcpPointToPlane::Rz(eulers[2]);
    Eigen::Vector3f translation(0.3, 0.5, 0);
    
    cout << endl << "Creating model with 10000 points ..." << endl;
    cout << "Creating template by specified transformtaion" << endl;
    
    int32_t k=0;
    for (int x=0; x<image_h; x++) {
        for (int y=0; y<image_w; y++) {
            
            double temp_x = (double)(x-image_h/2)*0.04;
            double temp_y = (double)(y-image_w/2)*0.04;
            double z=0.1+5*temp_x*exp(-temp_x*temp_x-temp_y*temp_y); // 0.1 offest to prevent 3D (0,0,0) point
            
            
            // project 2D to 3D            
            M[k*3+0] = (x-ox)/fx*z;
            M[k*3+1] = (y-oy)/fy*z;
            M[k*3+2] = z;
            
            Eigen::Vector3f temp_point(M[k*3+0], M[k*3+1], M[k*3+2]);
            temp_point = RR*temp_point;
            
            T[k*3+0] = temp_point[0] + translation[0];
            T[k*3+1] = temp_point[1] + translation[1];
            T[k*3+2] = temp_point[2] + translation[2];
            
            if(x==0 & y<=10)
            {
                printf("Model: %f %f %f\n", T[k*3+0], T[k*3+1], T[k*3+2]);
                printf("Input: %f %f %f\n", M[k*3+0], M[k*3+1], M[k*3+2]);
            }
            
            k++;
        }
    }
    
    // start with identity as initial transformation
    // in practice you might want to use some kind of prediction here
    Matrix R = Matrix::eye(3);
    Matrix t(3,1);
    
    // run point-to-plane ICP (-1 = no outlier threshold)
    cout << endl << "Running ICP (point-to-plane, no outliers)" << endl;  
    IcpPointToPlane icp(T,num,dim);
//     icp.fit(M,num,R,t,-1); //M*R+t = T
//     // results
//     cout << endl << "libicp result:" << endl;
//     cout << "R:" << endl << R << endl << endl;
//     cout << "t:" << endl << t << endl << endl;
    
    // use gauss newton based method
    Eigen::VectorXf x0(6,1); // (t_x, t_y, t_z, alpha_x, alpha_y, alpha_z)
    x0 << 0, 0, 0, 0, 0, 0;
    Eigen::Matrix<double, 6, 6> hessian;
//     icp.setCamParms(fx,fy,ox,oy);
//     icp.setImageSize(image_w, image_h);
    icp.use_normal = false;
    icp.use_projective = false;
    icp.fitGaussNewton(M, num, x0, -1);
    hessian = icp.getHessian();
    cout << "Hessian: uncertainty of the relative pose\n" << hessian << std::endl;
    
    printf("*****************************************\n");
    cout << "groundtruth \n";
    cout << "x: " << translation.transpose() << " " << eulers.transpose() << std::endl;
    cout << "R: \n" << RR << std::endl;
    cout << "t: \n" << translation.transpose() << std::endl;
    
    
//     // 2nd test
//     Eigen::Vector3f eulers2(-0.3, -0.3, 0.3);
//     RR = IcpPointToPlane::Rx(eulers2[0])*IcpPointToPlane::Ry(eulers2[1])*IcpPointToPlane::Rz(eulers2[2]);
//     Eigen::Vector3f translation2 = Eigen::Vector3f(-0.2, -0.1, 0.5);
//     k = 0;
//     for (int x=0; x<image_h; x++) {
//         for (int y=0; y<image_w; y++) {           
//             M[k*3+0] = T[k*3+0] - 0.5;
//             M[k*3+1] = T[k*3+1] - 0.3;
//             M[k*3+2] = T[k*3+2];
//             
//             k++;
//         }
//     }
//     
//     // start with identity as initial transformation
//     // in practice you might want to use some kind of prediction here
//     
//     // run point-to-plane ICP (-1 = no outlier threshold)
//     cout << endl << "Running ICP (point-to-plane, no outliers)" << endl; 
//     
//     // use gauss newton based method
//     x0 << 0, 0, 0, 0, 0, 0;
//     //     icp.setCamParms(fx,fy,ox,oy);
//     //     icp.setImageSize(image_w, image_h);
//     icp.use_normal = false;
//     icp.use_projective = false;
//     icp.fitGaussNewton(M, num, x0, -1);
//     hessian = icp.getHessian();
//     cout << "Hessian: uncertainty of the relative pose\n" << hessian << std::endl;
//     
//     printf("*****************************************\n");
//     cout << "groundtruth \n";
//     cout << "x: " << translation.transpose() << " " << eulers.transpose() << std::endl;
//     cout << "R: \n" << RR << std::endl;
//     cout << "t: \n" << translation.transpose() << std::endl;
    
    
    
    // free memory
    free(M);
    free(T);
    
    // success
    return 0;
}

