##Numerics:

diff_coef             = 1.0e4 #I had 5
hyperdiff_power       = 1.0

## Physics:

gravity               = 10.0
coriolis_parameter    = 1e-4;

### Convective Params

heating_amplitude     = 5.0e12 #originally 9 for heating, -8 for cooling
radiative_cooling     = (1.12/3.0)*1.0e-8
convective_timescale  = 28800.0
convective_radius     = 30000.0
critical_geopotential = 40.0
damping_timescale     = 2.0*86400.0
relaxation_height     = 39.0


### Simulation details
end_time = 6
