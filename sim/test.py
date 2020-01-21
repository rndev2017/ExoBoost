import sim
import planet as pl
import star as s
import matplotlib.pyplot as plt

n = 500

# Running simulations for star system similar to 24 Sex.
host_star = s.Star(1.12)
planet_b = pl.Planet(1.99, 452.8, 0.09)
planet_c = pl.Planet(0.86, 883, 0.29)

b_planet_params = planet_b.get_planet_params()
c_planet_params = planet_c.get_planet_params()


time = sim.create_length_of_observation(n)
phase_b = sim.calculate_phase(time, b_planet_params[1])
phase_c = sim.calculate_phase(time, c_planet_params[1])

k_b = sim.calcualte_K(b_planet_params[1]*24*60*60,
        host_star.get_star_mass(), b_planet_params[0], e=b_planet_params[2])

k_c = sim.calcualte_K(c_planet_params[1]*24*60*60,
        host_star.get_star_mass(), c_planet_params[0], e=c_planet_params[2])


sim_props_b = sim.create_sim_props(k_b, phase_b)
sim_props_c = sim.create_sim_props(k_c, phase_c)
sim_props = [sim_props_b, sim_props_c]


rv = sim.radvel(2, sim_props, n)
