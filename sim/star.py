sun_mass = 1.989e30 # Unites: kilograms


class Star:

    def __init__(self, mass):
        self.mass = mass


    def get_star_mass(self):
        return sun_mass*self.mass
