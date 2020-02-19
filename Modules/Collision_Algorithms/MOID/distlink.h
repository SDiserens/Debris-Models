/*
Copyright (c) 2018 R.V. Baluev and D.V. Mikryukov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef __DISTLINK_H__
#define __DISTLINK_H__

//This function finds suitable tolerances for functions calculating MOID.
template<typename realfp>
void detect_suitable_options(realfp& max_root_error,
                             realfp& min_root_error,
                             realfp& max_anom_error);

//This class is designed for orbit data storage.
//Functions set_data(), get_data() and get_*() can be used to access the orbital elements.
template<typename realfp>
class COrbitData
{
  template<typename T> friend class COrbitData;
  private:
    realfp a;           //semi-major axis
    realfp e;           //eccentricity
    realfp i;           //inclination
    realfp w;           //pericenter argument
    realfp Om;          //longitude of the ascending node
    realfp P[3];        //P vector components
    realfp Q[3];        //Q vector components
    void set_vectors(); //this function sets P and Q vector components

  public:
    COrbitData();
    COrbitData(realfp a_, realfp e_, realfp i_, realfp w_, realfp Om_);
    template<typename T> COrbitData(const COrbitData<T>& other);

    inline void set_data(realfp a_, realfp e_, realfp i_, realfp w_, realfp Om_) {
      a = a_; e = e_; i = i_; w = w_; Om = Om_;
      set_vectors();}
    inline void get_data(realfp& a_, realfp& e_, realfp& i_, realfp& w_, realfp& Om_) const {
      a_ = a; e_ = e; i_ = i; w_ = w; Om_ = Om;}

    inline realfp get_a() const {return a;}
    inline realfp get_e() const {return e;}
    inline realfp get_i() const {return i;}
    inline realfp get_w() const {return w;}
    inline realfp get_Om()const {return Om;}

    inline const realfp* vectorP() const {return P;}
    inline const realfp* vectorQ() const {return Q;}
    inline void get_vectors(realfp P_[3], realfp Q_[3]) const {
      P_[0] = P[0]; P_[1] = P[1]; P_[2] = P[2];
      Q_[0] = Q[0]; Q_[1] = Q[1]; Q_[2] = Q[2];}
};

/*
  This function allows to rapidly exclude from consideration
  those orbits that are too far from each other in the 3D space.
  It verifies simple (and therefore quick to check) but
  necessary condition of closeness two orbits in the space.

  Input:
    O1, O2  - data for two orbits;

    limit   - the maximum permissible difference between the
              corresponding apoapsis and periapsis.

  Returned value:
    true, if orbits O1 and O2 are not too far from each other;
    false if these are too far from each other and can be removed from consideration.
*/
template<typename realfp>
inline bool test_peri_apo(const COrbitData<realfp>& O1, const COrbitData<realfp>& O2, realfp limit)
{
 const realfp a1 = O1.get_a();
 const realfp e1 = O1.get_e();
 const realfp a2 = O2.get_a();
 const realfp e2 = O2.get_e();
 const realfp rp1= a1*(1-e1);
 const realfp ra1= a1*(1+e1);
 const realfp rp2= a2*(1-e2);
 const realfp ra2= a2*(1+e2);
 return (rp1-ra2<limit && rp2-ra1<limit);
}

//This structure is designed to represent all necessary information
//obtained during the computation of the MOID.
template<typename realfp>
struct SMOIDResult
{
 bool good;                   //true if the result is reliable
 realfp distance;             //minimum distance between orbits
 realfp distance_error;       //numeric uncertainty of the distance
 realfp u1;                   //eccentric anomaly on the first orbit
 realfp u1_error;             //its numeric uncertainty
 realfp u2;                   //eccentric anomaly on the second orbit
 realfp u2_error;             //its numeric uncertainty
 unsigned short root_count;   //number of real roots of the function g(u1)
 realfp min_delta;            //the minimum quantity delta among all non-real roots - see description in the paper
 unsigned int iter_count;     //number of Newtonian iterations of g(u), sum for all 16 roots
 unsigned int iter_count_2D;  //number of Newtonian 2D iterations of rho(u,u')
 realfp time;                 //CPU time used (in seconds), actually appears unreliable
                              //in modern hardware (millisecond accuracy appears not enough)
 SMOIDResult();
};

//This structure is designed for storage two linking coefficients
//and the mutual inclination of the pair of orbits.
template<typename realfp>
struct SLCResult
{
 realfp I;                    //mutual inclination
 realfp l;                    //the first or the third linking coefficient; it depends on I
 realfp lmod;                 //modified first linking coefficient, |lmod| is always smaller than |l|, and is a good upper limit on MOID^2
 realfp l2;                   //the continuous second linking coefficient; it is calculated for every pair of orbits,
                              //regardless of whether the mutual inclination I is small (however small) or not
 SLCResult();
};

/*
  The function below calculates two linking coefficients
  and the mutual inclination of the pair of orbits.

  Input:
    O1, O2           - data for the pair of orbits;

    min_mut_incl     - threshold value of the mutual inclination I
                       that determines whether we will calculate the
                       first (I>=min_mut_incl) or the third (I<min_mut_incl)
                       linking coefficient in l.
*/
template<typename realfp>
SLCResult<realfp> LC(const COrbitData<realfp>& O1, const COrbitData<realfp>& O2, realfp min_mut_incl);

/*
  The function below calculates MOID by means of finding critical points of the squared distance.

  Input:
    O1, O2           - data for the pair of orbits;

    max_root_error   - maximum relative error allowed for the complex roots of the polynomial P(z);
                       also used to self-diagnose unreliable result (the flag result.good);

    min_root_error   - desirable precision of the roots of P(z); it is safe to set it to zero, but
                       on many CPUs this may result in a side effect of excessive iteration of the
                       roots until the long double 80-bit accuracy even if realfp was double;

    nu               - this is a scaling factor for all error estimations;
                       typically should be set to unit.

  Return value: SMOIDResult record.
*/
template<typename realfp>
SMOIDResult<realfp> MOID_fast(const COrbitData<realfp>& O1, const COrbitData<realfp>& O2,
                              realfp max_root_error, realfp min_root_error,
                              realfp nu=static_cast<realfp>(1));

/*
  The function below calculates MOID by means of one-dimensional search of the minimum of
  the distance. Only the first eccentric anomaly is scanned; the position on the second
  orbit is calculated analytically. This function can be used when MOID_fast returns false
  in result.good.

  Input:
    O1,O2           - data for the pair of orbits;

    densities       - array of positive integers terminated by zero; it determines how the
                      search segments will be split during the calculation;

                      E.g.: densities={1000, 30, 3, 0}. First, split the full [0, 2pi] range
                      into 1000 equal segments. Select two neighbour segments where the minimum
                      MOID is located. Then split these two segmets into 30 equal subsegments,
                      thus reducing the grid step to 1/15 of the previous value. Then every
                      pair of subsequent segments with minimum MOID will be trisected (so step
                      reduction factor is 2/3), until the required precision is reached;

                      The last non-zero element of densities[] is recommended to be 4 (this
                      corresponds to the bisection method). Non-zero elements less than 3
                      are meaningless and forced to be 3;

    max_dist_error  - maximum error allowed in the distance; it is used to determine when
                      the algorithm should stop;

    max_anom_error  - maximum error allowed for determination of the eccentric anomaly
                      being scanned; it is used to determine when the algorithm should
                      stop;

  Return value: SMOIDResult record.

  The algorithm stops when the first eccentric anomaly and the minimum distance are both
  located with required accuracies.

  NB: As of the stable 1.0 version, this function was optimized greatly. It scans only a reduced
      range near orbital nodes (neglecting the portions where MOID cannot be located). Also, it
      automatically selects the orbits order (O1,O2) or (O2,O1) to obtain the smallest possible
      scan range. In case if this range appears equal for the both (e.g. if it appeared to be
      [0,2pi]) then the user-supplied order is preserved.
*/
template<typename realfp>
SMOIDResult<realfp> MOID_direct_search(const COrbitData<realfp>& O1, const COrbitData<realfp>& O2,
                               const unsigned int densities[],
                               realfp max_dist_error, realfp max_anom_error);

#endif
