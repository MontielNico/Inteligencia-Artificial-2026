import entornos_o
from random import choice
import random

class DosCuartosEstocástico(entornos_o.Entorno):

    #Probabilidad de que el agente limpie correctamente
    probabilidad_limpieza = 0.8
    probablidad_suciedad = 0.2

    #Probabilidad de que el agente se mueva de habitación correctamente
    probabilidad_movimiento = 0.9
    probabilidad_no_movimiento = 0.1

    def __init__(self, x0=[choice(["A", "B"]), choice(["sucio", "limpio"]), choice(["sucio", "limpio"])]):
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
            if random.random() < self.probabilidad_limpieza:
                self.x[" AB".find(self.x[0])] = "limpio"
                self.desempeño -= 0.5
            else:
                self.x[" AB".find(self.x[0])] = "sucio"
                self.desempeño -= 1
                print("Fallo al limpiar")
        elif acción == "ir_A":
            if random.random() < self.probabilidad_movimiento:
                self.x[0] = "A"
                self.desempeño -= 1
            else:
                self.x[0] = "B"
                self.desempeño -= 1
                print("Fallo al ir a A")
        elif acción == "ir_B":
            if random.random() < self.probabilidad_movimiento:
                self.x[0] = "B"
                self.desempeño -= 1
            else:
                self.x[0] = "A"
                self.desempeño -= 1
                print("Fallo al ir a B")
    
    def percepción(self):
        return self.x # Entorno totalmente observable

class AgenteBasadoUtilidadDosCuartos(entornos_o.Agente):
    def __init__(self):
        self.utilidad = {
            ("A", "limpio", "limpio"): 100,
            ("A", "limpio", "sucio"): 40,  # Peor porque estamos lejos de la basura en B
            ("A", "sucio", "limpio"): 60,  # Mejor porque estamos parados donde hay basura
            ("A", "sucio", "sucio"): 0,
            ("B", "limpio", "limpio"): 100,
            ("B", "limpio", "sucio"): 60,  # Mejor porque estamos parados donde hay basura
            ("B", "sucio", "limpio"): 40,  # Peor porque estamos lejos de la basura en A
            ("B", "sucio", "sucio"): 0,
        }
    
    def programa(self, percepcion):
        robot, a, b = percepcion
        
        acciones = ["nada", "limpiar", "ir_A", "ir_B"]
        max_ut = -float('inf')
        mejor_accion = "nada"
        
        # El agente evalúa la utilidad esperada de cada acción
        for accion in acciones:
            ut = 0
            if accion == "limpiar":
                if robot == "A":
                    ut = 0.8 * self.utilidad[("A", "limpio", b)] + 0.2 * self.utilidad[("A", "sucio", b)]
                else:
                    ut = 0.8 * self.utilidad[("B", a, "limpio")] + 0.2 * self.utilidad[("B", a, "sucio")]
            elif accion == "ir_A":
                ut = 0.9 * self.utilidad[("A", a, b)] + 0.1 * self.utilidad[("B", a, b)]
            elif accion == "ir_B":
                ut = 0.9 * self.utilidad[("B", a, b)] + 0.1 * self.utilidad[("A", a, b)]
            elif accion == "nada":
                ut = self.utilidad[(robot, a, b)]
                
            # Selecciona la acción que maximice la utilidad esperada
            if ut > max_ut:
                max_ut = ut
                mejor_accion = accion
                
        return mejor_accion

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
    print(DosCuartosEstocástico().x)
    entornos_o.simulador(DosCuartosEstocástico(), AgenteBasadoUtilidadDosCuartos(), 10)
    entornos_o.simulador(DosCuartosEstocástico(), AgenteAleatorio(acciones), 10)

if __name__ == "__main__":
    test()