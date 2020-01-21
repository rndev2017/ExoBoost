jup_mass = 1.898e27 # Units: kilograms


class Planet:

    def __init__(self, mass, period, eccentricty):
        self.mass = mass
        self.period = period
        self.eccentricty = eccentricty


    def get_planet_params(self):
        return [self.mass*jup_mass, self.period, self.eccentricty]
