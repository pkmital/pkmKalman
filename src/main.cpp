#include <iostream>
#include "pkmKalman.h"

void testKalmanFilter2D()
{
    float F[2][2] = { { 1, 1 },
                      { 0, 1 } };
    float H[2] = {1, 0};
    
    // init kalman filter
    pkm::Kalman kf;
    
    // track position and velocity (assumes acceleration is 0)
    kf.allocate(2, 1);
    
    // set initial pos/vel to 0
    // (x)
    kf.setState(pkm::Mat(2, 1, 0.0f));
    
    // create a transition matrix for modeling the physics of the system
    // the first row corresponds to using the velocity and position to update the new position:
    // x = 1*x + 1*dx*dt
    // the second row uses only the velocity to update the velocity:
    // dx = 0*x + 1*dx*dt
    // (F)
    kf.setStateTransitionMatrix(pkm::Mat(2, 2, &F[0][0]));
    
    // How much noise we expect to have in our measurement sensor (fixed gaussian)
    // (R)
    kf.setMeasurementNoise(0.1f);
    
    // Large initial covariance of starting pos/vel saying how much certainty we have in our state (variable)
    // (P)
    kf.setStateCovariance(100.0f);
    
    // Unmodeled noise
    // (Q)
    kf.setProcessNoise(0.1, 0.01);
    
    // Converts our state to a measurement, so position, (what we measure), = x*1 + dx*0
    // (H)
    kf.setMeasurementFunction(pkm::Mat(1, 2, &H[0]));
    
    size_t n_obs = 100;
    float noise = 10.0;
    pkm::Mat m_z(1, 2, 0.0f);
    m_z[1] = 100.0f;
    m_z.interpolate(1, n_obs);
    m_z = m_z + pkm::Mat::rand(1, n_obs) * noise;
    for (size_t i = 0; i < n_obs; i++) {
        kf.update(m_z.colRange(i, i+1));
        kf.predict();
        std::cout << "obs: " << m_z.colRange(i, i+1)[0] << ", filt: " << kf.getState()[0] << " cov: " << kf.getStateCovariance()[0] << endl;
    }
}


void testKalmanFilter3D()
{
    float F[3][3] = { { 1, 1, 1 },
                      { 0, 1, 1 },
                      { 0, 0, 1 } };
    float H[3] = {1, 0, 0};
    
    // init kalman filter
    pkm::Kalman kf;
    
    // track position and velocity, and acceleration (assumes torque is 0)
    kf.allocate(3, 1);
    
    // set initial pos/vel/accel to 0
    // (x)
    kf.setState(pkm::Mat(3, 1, 0.0f));
    
    // create a transition matrix for modeling the physics of the system
    // the first row corresponds to using the velocity and position to update the new position:
    // x = 1*x + 1*dx*dt + 1*dx^2*d^2t
    // the second row uses the velocity and acceleration to update the velocity
    // dx = 0*x + 1*dx*dt + 1*dx^2*d^2t
    // dx^2 = 0*x + 0*dx*dt + 1*dx^2*d^2t
    // (F)
    kf.setStateTransitionMatrix(pkm::Mat(3, 3, &F[0][0]));
    
    // How much noise we expect to have in our measurement sensor (fixed gaussian)
    // (R)
    kf.setMeasurementNoise(0.1f);
    
    // Large initial covariance of starting pos/vel saying how much certainty we have in our state (variable)
    // (P)
    kf.setStateCovariance(100.0f);
    
    // Unmodeled noise
    // (Q)
    kf.setProcessNoise(0.1, 0.01);
    
    // Converts our state to a measurement, so position, (what we measure), = x*1 + dx*0 + dx^2*0
    // (H)
    kf.setMeasurementFunction(pkm::Mat(1, 3, &H[0]));
    
    size_t n_obs = 1000;
    float noise = 1.0;
    pkm::Mat m_z(1, 2, 0.0f);
    m_z[1] = 100.0f;
    m_z.interpolate(1, n_obs);
    m_z = m_z + pkm::Mat::rand(1, n_obs) * noise;
    for (size_t i = 0; i < n_obs; i++) {
        kf.update(m_z.colRange(i, i+1));
        kf.predict();
        std::cout << "obs: " << m_z.colRange(i, i+1)[0] << ", filt: " << kf.getState()[0] << endl;
    }
}

int main(int argc, const char * argv[])
{
    testKalmanFilter2D();
    testKalmanFilter3D();
    return 0;
}
