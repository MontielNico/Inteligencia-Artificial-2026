import numpy as np
from habitacion import Habitacion

rng = np.random.default_rng()


def limpiar(habitacion):
    if habitacion.estadoHabitacion == "Sucio": 
        print("Habitación sucia, limpiando...")
        print("Listo! yendo a la otra habitación")
        habitacion.estadoHabitacion = "Limpio"
    else:
        print("La habitación ya está limpia, cambiando de habitacion")


def limpiarHabitaciones():
    numeroHabitacion = rng.integers(low=1, high=3) #numpy no incluye el ultimo numero, por eso es 3
    habitacion1 = Habitacion(1)
    habitacion2 = Habitacion(2)

    print("Estado inicial: ")
    print("Habitación 1: " + habitacion1.estadoHabitacion)
    print("Habitación 2: " + habitacion2.estadoHabitacion)
    print("Agente en habitación: " + str(numeroHabitacion))
    print("____________________________")

    for i in range(20):
        print("Habitación: " + str(numeroHabitacion))
        if numeroHabitacion == 1 : 
            limpiar(habitacion1)
            numeroHabitacion = 2
        else:
            limpiar(habitacion2)
            numeroHabitacion = 1
        print("____________________________")



print(limpiarHabitaciones())