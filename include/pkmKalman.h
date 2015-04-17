/*
 *  pkmKalman.h
 *
 
 Kalman Filtering
 
 Copyright (C) 2015 Parag K. Mital
 
 The Software is and remains the property of Parag K Mital
 ("pkmital") The Licensee will ensure that the Copyright Notice set
 out above appears prominently wherever the Software is used.
 
 The Software is distributed under this Licence:
 
 - on a non-exclusive basis,
 
 - solely for non-commercial use in the hope that it will be useful,
 
 - "AS-IS" and in order for the benefit of its educational and research
 purposes, pkmital makes clear that no condition is made or to be
 implied, nor is any representation or warranty given or to be
 implied, as to (i) the quality, accuracy or reliability of the
 Software; (ii) the suitability of the Software for any particular
 use or for use under any specific conditions; and (iii) whether use
 of the Software will infringe third-party rights.
 
 pkmital disclaims:
 
 - all responsibility for the use which is made of the Software; and
 
 - any liability for the outcomes arising from using the Software.
 
 The Licensee may make public, results or data obtained from, dependent
 on or arising out of the use of the Software provided that any such
 publication includes a prominent statement identifying the Software as
 the source of the results or the data, including the Copyright Notice
 and stating that the Software has been made available for use by the
 Licensee under licence from pkmital and the Licensee provides a copy of
 any such publication to pkmital.
 
 The Licensee agrees to indemnify pkmital and hold them
 harmless from and against any and all claims, damages and liabilities
 asserted by third parties (including claims for negligence) which
 arise directly or indirectly from the use of the Software or any
 derivative of it or the sale of any products based on the
 Software. The Licensee undertakes to make no liability claim against
 any employee, student, agent or appointee of pkmital, in connection
 with this Licence or the Software.
 
 
 No part of the Software may be reproduced, modified, transmitted or
 transferred in any form or by any means, electronic or mechanical,
 without the express permission of pkmital. pkmital's permission is not
 required if the said reproduction, modification, transmission or
 transference is done without financial return, the conditions of this
 Licence are imposed upon the receiver of the product, and all original
 and amended source code is included in any transmitted product. You
 may be held legally responsible for any copyright infringement that is
 caused or encouraged by your failure to abide by these terms and
 conditions.
 
 You are not permitted under this Licence to use this Software
 commercially. Use for which any financial return is received shall be
 defined as commercial use, and includes (1) integration of all or part
 of the source code or the Software into a product for sale or license
 by or on behalf of Licensee to third parties or (2) use of the
 Software or any derivative of it for research with the final aim of
 developing software products for sale or license to a third party or
 (3) use of the Software or any derivative of it for research with the
 final aim of developing non-software products for sale or license to a
 third party, or (4) use of the Software to provide any service to an
 external organisation for which payment is received. If you are
 interested in using the Software commercially, please contact pkmital to
 negotiate a licence. Contact details are: parag@pkmital.com
 
 */

 /*
 
 Written by Parag K. Mital, 2015
 
 Based off of Python code from:
 https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
 (Kalman and Bayesian Filters in Python by Roger R. Labbe is licensed under a Creative Commons Attribution 4.0 International License.)
 
 */

#include "pkmMatrix.h"

using namespace pkm;

namespace pkm {
    
    //----------------------------------------------------------------------
    /// \class pkm::Kalman defines an n-dimensional Kalman filter
    //----------------------------------------------------------------------
    class Kalman
    {
    public:
        //----------------------------------------------------------------------
        Kalman()
        {
            
        }
        
        //----------------------------------------------------------------------
        /*
         \brief Initializes all internal variables based on the size of the state/measurement/control variables
         */
        //----------------------------------------------------------------------
        void allocate(int sx,        ///< number of state variables
                      int sz,        ///< number of measurement inputs
                      int su = 0)    ///< number of control inputs
        {
            dim_x = sx;
            dim_z = sz;
            dim_u = su;
            
            x = Mat::zeros(dim_x, 1);
            P = Mat::eye(dim_x);
            Q = Mat::eye(dim_x);
            u = Mat::zeros(dim_x, 1);
            B = Mat::Mat();
            F = Mat::Mat();
            H = Mat::Mat();
            Ht = Mat::Mat();
            R = Mat::eye(dim_z);
            I = Mat::eye(dim_x);
        }
        
        //----------------------------------------------------------------------
        void update(Mat Z)
        {
            update(Z, Mat());
        }
        
        //----------------------------------------------------------------------
        void update(Mat Z,      ///< new measurement for this update
                    Mat R)      ///< overrides the measure noise
        {
            if (Z.isEmpty()) {
                printf("[pkm::Kalman]::Warning Z is empty!\n");
                return;
            }
            
            if (R.isEmpty()) {
                R = this->R;
            }
            else if (R.size() == 1) {
                float r = R[0];
                R = Mat::identity(dim_z);
                R.multiply(r);
            }
            
            // Compute error (residual) between measurement and prediction
            // y = z - Hx
            Mat y = Z - H.dot(x);
            
            // Project system uncertainty into Kalman gain
            // S = HPH' + R
            Mat S = H.dot(P).dot(Ht) + R;
            
            // Map system uncertainty into Kalman gain
            // K = PH'inv(S)
            Mat K = P.dot(Ht).dot(S.getInv());
            
            // Predict new x using the residual scaled by Kalman gain matrix
            // x = x + Ky
            x = x + K.dot(y);
            
            // P = (I-KH)P(I-KH)' + KRK'
            Mat I_KH = I - K.dot(H);
            P = I_KH.dot(P).dot(I_KH.getTranspose()) + K.dot(R).dot(K.getTranspose()); //- P.dot(H.getTranspose()).dot(K.getTranspose()) + K.dot(S).dot(K.getTranspose());

        }
        
        //----------------------------------------------------------------------
        void predict(Mat u)
        {
            if (u.isEmpty()) {
                predict();
            }
            else {
                x = F.dot(x) + B.dot(u);
                P = F.dot(P).dot(F.getTranspose()) + Q;
            }
        }
        
        //----------------------------------------------------------------------
        void predict()
        {
            x = F.dot(x);
            P = F.dot(P).dot(F.getTranspose()) + Q;
        }
        
    private:
        
        
        //----------------------------------------------------------------------
        
        /*
         \brief The Number of state variables for the Kalman filter;
         \brief e.g. tracking 2d position/velocity requires 4 vairables: x, dx, y, dy
        */
        int dim_x;
        
        /*
         \brief The number of measurement inputs. 
         \brief e.g. if the sensor provides position (x,y), then 2.
        */
        int dim_z;
        
        /*
         \brief (optional)
         \brief The number of control inputs (if used)
         \brief 0 indicates it is not used.
        */
        int dim_u;
        
    public:
        
        //----------------------------------------------------------------------
        Mat getState()
        {
            return x;
        }
        Mat getX()
        {
            return x;
        }
        void setState(const Mat &x)
        {
            set_x(x);
        }
        void set_x(const Mat &x)
        {
            if (this->x.rows != x.rows && this->x.cols != x.cols)
            {
                printf("[pkm::Kalman]::ERROR! dimensions of x must match dimensions set during initialization!\n");
                return;
            }
            this->x = x;
        }
        
        //----------------------------------------------------------------------
        Mat getStateCovariance()
        {
            return getP();
        }
        Mat getP()
        {
            return P;
        }
        void setStateCovariance(const Mat &P)
        {
            set_P(P);
        }
        void setStateCovariance(float P)
        {
            set_P(Mat::identity(dim_x) * P);
        }
        void set_P(const Mat &P)
        {
            if (this->P.rows != P.rows && this->P.cols != P.cols) {
                printf("[pkm::Kalman]::ERROR! dimensions of P must match dimensions set during initialization!\n");
                return;
            }
            this->P = P;
        }
        
        //----------------------------------------------------------------------
        void setStateTransitionMatrix(const Mat &F)
        {
            set_F(F);
        }
        void set_F(const Mat &F)
        {
            if (F.rows != dim_x && F.cols != dim_x) {
                printf("[pkm::Kalman]::ERROR! dimensions of F must match dimensions set during initialization (F should be %d x %d)!\n", dim_x, dim_x);
                return;
            }
            this->F = F;
        }
        
        //----------------------------------------------------------------------
        void setMeasurementFunction(const Mat &H)
        {
            set_H(H);
        }
        void set_H(const Mat &H)
        {
            if (H.rows != dim_z && H.cols != dim_x) {
                printf("[pkm::Kalman]::ERROR! dimensions of H must match dimensions set during initialization (H should be %d x %d)!\n", dim_z, dim_x);
                return;
            }
            this->H = H;
            Ht = H.getTranspose();
        }
        
        //----------------------------------------------------------------------
        void setMeasurementNoise(const Mat &R)
        {
            set_R(R);
        }
        void setMeasurementNoise(float R)
        {
            set_R(Mat::identity(dim_z) * R);
        }
        void set_R(const Mat &R)
        {
            if (this->R.rows != R.rows && this->R.cols != R.cols) {
                printf("[pkm::Kalman]::ERROR! dimensions of R must match dimensions set during initialization!\n");
                return;
            }
            this->R = R;
        }
        
        //----------------------------------------------------------------------
        void setProcessNoise(float dt,      // time step in whatever unit the filter uses
                             float var)     // noise variance
        {
            if (dim_x == 2) {
                float Qdat[2][2] = {{ powf(.25*dt,4), powf(.5*dt,3) },
                                 { powf(.5*dt,3),  powf(dt,2)    }};
                Mat Q(2, 2, &Qdat[0][0]);
                Q.multiply(var);
                setProcessNoise(Q);
                
            }
            else if(dim_x == 3) {
                float Qdat[3][3] = {{ powf(.25*dt,4), powf(.5*dt,3), powf(.5*dt,2) },
                                 { powf(.5*dt,3),    powf(dt,2),       dt      },
                                 { powf(.5*dt,2),       dt,            1.0f      }};
                Mat Q(3, 3, &Qdat[0][0]);
                Q.multiply(var);
                setProcessNoise(Q);
            }
            else {
                printf("[pkm::Kalman]::ERROR! Setting process noise using dt/var formula only supports dim = 2 or 3!  Manually set Process Noise using another formulation (e.g. setProcessNoise(const Mat &Q))\n");
                return;
            }
            
        }
        void setProcessNoise(const Mat &Q)
        {
            set_Q(Q);
        }
        void set_Q(const Mat &Q)
        {
            if (this->Q.rows != Q.rows && this->Q.cols != Q.cols) {
                printf("[pkm::Kalman]::ERROR! dimensions of Q must match dimensions set during initialization!\n");
                return;
            }
            this->Q = Q;
        }
        
        
        
    private:
        
        //----------------------------------------------------------------------
        
        /*
         \brief The state
        */
        Mat x;
        
        /*
         \brief Uncertainty covariance
        */
        Mat P;
        
        /*
         \brief The process uncertainty
        */
        Mat Q;
        
        /*
         \brief The motion vector
        */
        Mat u;
        
        /*
         \brief The control transition matrix
        */
        Mat B;
        
        /*
         \brief The state transition matrix
        */
        Mat F;

        /*
         \brief The measurement function
        */
        Mat H, Ht;
        
        /*
         \brief The state uncertainty from measurement
        */
        Mat R;

        /*
         \brief Identity matrix
        */
        Mat I;
        
        //----------------------------------------------------------------------

    };
    
}