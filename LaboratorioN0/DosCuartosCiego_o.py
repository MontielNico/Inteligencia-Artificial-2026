from random import choice
import entornos_o
class DosCuartosCiego(entornos_o.Entorno):
    def __init__(self, x0=[choice(["A","B"]), choice(["sucio","limpio"]), choice(["sucio","limpio"])]):
        self.x = x0[:]
        self.desempeño = 0
    
    def acción_legal(self, acción):
        return acción in ("ir_A", "ir_B", "limpiar", "nada")
    
    def transición(self, acción):
        if not self.acción_legal(acción):
            raise ValueError("La acción no es legal para este estado")
        
        robot, a, b = self.x
        if a == "sucio" or b == "sucio":
            self.desempeño -= 1
        if acción == "limpiar":
            self.x[" AB".find(self.x[0])] = "limpio"
            self.desempeño -= 0.5
        elif acción == "ir_A":
            self.x[0] = "A"
            self.desempeño -= 1
        elif acción == "ir_B":
            self.x[0] = "B"
            self.desempeño -= 1
    
    def percepción(self):
        return self.x[0] # Entorno parcialmente observable

class AgenteModeloDosCuartosCiego(entornos_o.Agente):
    def __init__(self):
        self.modelo = ["A", "sucio", "sucio"] # Modelo inicial donde el agente se situa en la peor situacion posible
    
    def programa(self, percepción):
        robot = percepción
        self.modelo[0] = robot

        # Evaluamos si alguna de las dos habitaciones en nuestro modelo sigue sucia
        if self.modelo[1] == "sucio" or self.modelo[2] == "sucio":
            if self.modelo[" AB".find(robot)] == "sucio":
                self.modelo[" AB".find(robot)] = "limpio"
                return "limpiar"
            elif self.modelo[0] == "A":
                return "ir_B"
            else:
                return "ir_A"
        else:
            return "nada"

class AgenteAleatorio(entornos_o.Agente):
    """
    Un agente que solo regresa una accion al azar entre las acciones legales

    """
    def __init__(self, acciones):
        self.acciones = acciones

    def programa(self, percepción):
        return choice(self.acciones)
        

def test():
    acciones = ["ir_A", "ir_B", "limpiar", "nada"]
    print("Estado inicial")
    print(DosCuartosCiego().x)
    entornos_o.simulador(DosCuartosCiego(), AgenteModeloDosCuartosCiego(), 10)
    entornos_o.simulador(DosCuartosCiego(), AgenteAleatorio(acciones), 10)

if __name__ == "__main__":
    test()