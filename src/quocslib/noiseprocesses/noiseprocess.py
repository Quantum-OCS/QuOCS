class NoiseProcess:
    """
    Base class for noise processes.
    """
    def __init__(self, noise_type, dt):
        self.noise_type = noise_type
        self.dt = dt
