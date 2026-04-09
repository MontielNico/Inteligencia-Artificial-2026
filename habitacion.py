import numpy as np

class Habitacion:
    def __init__(self, numero):
        rng = np.random.default_rng()
        
        self.numeroHabitacion = numero
        
        self.estadoHabitacion = rng.choice(["Limpio", "Sucio"])

    def mostrar_info(self):
        return f"Habitación {self.numeroHabitacion}: Estado -> {self.estadoHabitacion}"

    

