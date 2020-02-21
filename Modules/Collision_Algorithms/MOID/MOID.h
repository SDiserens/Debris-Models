#define ELEMENTS struct elements

#pragma pack(4)
ELEMENTS
{
	double perih_time, q, ecc, incl, arg_per, asc_node;
double epoch,  mean_anomaly;
/* derived quantities: */
double lon_per, minor_to_major;
double perih_vec[3], sideways[3];
double angular_momentum, major_axis, t0, w0;
double abs_mag, slope_param, gm;
int is_asteroid, central_obj;
};
#pragma pack( )

typedef struct
{
	double obj1_true_anom, jd1;       /* these are set in find_moid_full */
	double obj2_true_anom, jd2;       /* NOT SET YET */
	double barbee_speed;              /* in AU/day */
} moid_data_t;

void vector_cross_product(double *xprod, const double *a, const double *b);

double dot_product(const double *a, const double *b);

double vector3_length(const double *vect);

void derive_quantities(ELEMENTS *e, const double gm);

void setup_orbit_vectors(ELEMENTS *e);

double point_to_ellipse(const double a, const double b,
	const double x, const double y, double *dist);

double find_moid_full(OrbitalElements &primary, OrbitalElements &secondary, moid_data_t *mdata);
double find_moid_full(const ELEMENTS *elem1, const ELEMENTS *elem2, moid_data_t *mdata);
