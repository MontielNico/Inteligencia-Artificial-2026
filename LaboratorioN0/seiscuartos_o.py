import entornos_o 
from random import choice

class SeisCuartos(entornos_o.Entorno):
    def __init__(self, x0=["A", "sucio", "sucio", "sucio", "sucio", "sucio", "sucio"]):
        self.x = x0[:]
        self.desempeño = 0
    
    def acción_legal(self, acción):
        robot = self.x[0]

        if acción == "subir":
            return robot in ("A", "C") #A y C son las habitaciones de los extremos de abajo donde se encuentran las escaleras
        elif acción == "bajar":
            return robot in ("E") #E es la habitacion del centro de arriba donde se encuentra la escalera   
        elif acción == "ir_Derecha":
            return robot in ("A", "B", "D", "E") #Excluye F y C por que son los extremos derechos
        elif acción == "ir_Izquierda":
            return robot in ("B", "C", "E", "F") #Excluye A y D por que son los extremos izquierdos
        elif acción == "limpiar":
            return True
        elif acción == "nada":
            return True
        else:
            return False
    
    def transición(self, acción):
        if not self.acción_legal(acción):
            raise ValueError("La acción no es legal para este estado")

        robot, a, b, c, d, e, f = self.x

        #Penalización si quedan habitaciones sucias
        if a == "sucio" or b == "sucio" or c == "sucio" or d == "sucio" or e == "sucio" or f == "sucio":
            self.desempeño -= 1
        
        #Costos de las acciones
        #Limpiar tiene el menor costo que cualquier otra acción
        if acción == "limpiar":
            self.x[" ABCDEF".find(robot)] = "limpio"
            self.desempeño -= 0.5

        #Moverse a la derecha tiene un costo de 1
        elif acción == "ir_Derecha":
            mapa_derecha = {"A": "B", "B": "C", "D": "E", "E": "F"}
            self.desempeño -=1
            self.x[0] = mapa_derecha[robot]

        #Moverse a la izquierda tiene un costo de 1
        elif acción == "ir_Izquierda":
            mapa_izquierda = {"B": "A", "C": "B", "E": "D", "F": "E"}
            self.desempeño -=1
            self.x[0] = mapa_izquierda[robot]

        #Moverse hacia arriba tiene un costo de 2
        elif acción == "subir":
            mapa_subir = {"A": "D", "C": "F"}
            self.desempeño -=2
            self.x[0] = mapa_subir[robot]

        #Moverse hacia abajo tiene un costo de 2
        elif acción == "bajar":
            mapa_bajar = {"E": "B"}
            self.desempeño -=2
            self.x[0] = mapa_bajar[robot]
    
    def percepción(self):
        return self.x[0], self.x[' ABCDEF'.find(self.x[0])]


class AgenteReactivoModelosSeisCuartos(entornos_o.Agente):
    def __init__(self):
        self.modelo = ["A", "sucio", "sucio", "sucio", "sucio", "sucio", "sucio"]
        
    
    def programa(self, percepción):
        robot, situación = percepción

        self.modelo[0] = robot
        self.modelo[' ABCDEF'.find(robot)] = situación

        a, b, c, d, e, f = self.modelo[1], self.modelo[2], self.modelo[3], self.modelo[4], self.modelo[5], self.modelo[6]

        if a == b == c == d == e == f == "limpio":
            return "nada"
        elif situación == "sucio":
            return "limpiar"
        elif robot == "A":
            # Si arriba es donde hay suciedad, subimos, de lo contrario nos barremos por abajo
            if d == "sucio" or e == "sucio":
                return "subir"
            else:
                return "ir_Derecha"
                
        elif robot == "B":
            # Si la suciedad está a la derecha o arriba a la derecha, avanza
            if c == "sucio" or f == "sucio" or e == "sucio":
                return "ir_Derecha"
            else:
                # Solo quedan A o D sucios atrás
                return "ir_Izquierda"
                
        elif robot == "C":
            # Si cualquier parte del piso superior está sucio, la vía más de acceso es subir a F
            if f == "sucio" or e == "sucio" or d == "sucio":
                return "subir"
            else:
                # Si estamos limpios arriba, retrocedemos por abajo buscando a B o A
                return "ir_Izquierda"
                
        elif robot == "D":
            # Desde D (esquina) solo podemos ir al centro E
            return "ir_Derecha"
            
        elif robot == "E":
            if f == "sucio":
                return "ir_Derecha"
            elif d == "sucio":
                return "ir_Izquierda"
            else:
                # Si arrriba estamos limpios, bajamos a inspeccionar la planta baja
                return "bajar"
                
        elif robot == "F":
            # Desde F (esquina) solo podemos ir al centro E
            return "ir_Izquierda"

class AgenteAleatorioSeisCuartos(entornos_o.Agente):
    def __init__(self, acciones):
        self.acciones = acciones
    
    def programa(self, percepcion):
        cuarto_actual = percepcion[0]
        # Empezamos asumiendo que "limpiar" y "nada" y la lista de acciones base se filtran según el cuarto
        acciones_legales = ["limpiar", "nada"]
        
        if cuarto_actual in ("A", "C") and "subir" in self.acciones:
            acciones_legales.append("subir")
        if cuarto_actual == "E" and "bajar" in self.acciones:
            acciones_legales.append("bajar")
        if cuarto_actual in ("A", "B", "D", "E") and "ir_Derecha" in self.acciones:
            acciones_legales.append("ir_Derecha")
        if cuarto_actual in ("B", "C", "E", "F") and "ir_Izquierda" in self.acciones:
            acciones_legales.append("ir_Izquierda")
            
        return choice(acciones_legales)

def test():

    acciones = ["ir_Derecha", "ir_Izquierda", "subir", "bajar", "limpiar", "nada"]

    print("Prueba del entorno con un agente reactivo con modelo")
    entornos_o.simulador(SeisCuartos(), AgenteReactivoModelosSeisCuartos(), 100)
    print("Prueba del entorno con un agente aleatorio")
    entornos_o.simulador(SeisCuartos(), AgenteAleatorioSeisCuartos(acciones), 100)

if __name__ == "__main__":
    test()
    