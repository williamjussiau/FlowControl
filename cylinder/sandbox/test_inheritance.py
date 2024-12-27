from enum import Enum

class SOLVER(Enum):
    DEFAULT = 0
    LU = 1
    KRYLOV = 2

class Parent():
    def __init__(self, solver_type=None):
        self.solver = self.make_solver(type=solver_type)

    def make_solver(self, type=None):
        """returns a solver, do not assign"""
        return dict(type=SOLVER.DEFAULT) # default solver
    
class Child(Parent):
    def __init__(self, solver_type=None):
        super().__init__(solver_type=solver_type)

    def make_solver(self, type=None):
        if type is None or type==SOLVER.DEFAULT:
            solver = super().make_solver()
        elif type==SOLVER.LU:
            solver = dict(type=SOLVER.LU) # other solver, eg LU...
        elif type==SOLVER.KRYLOV:
            solver = dict(type=SOLVER.KRYLOV) # other solver, eg Krylov...
        else:
            raise ValueError("Unknown solver code")
        return solver

if __name__ == "__main__":
    C1 = Child(solver_type=SOLVER.LU)
    C2 = Child(solver_type=SOLVER.KRYLOV)
    C3 = Child(solver_type=SOLVER.DEFAULT)
    C4 = Child()

    print("C1 solver: ", C1.solver)
    print("C2 solver: ", C2.solver)
    print("C3 solver: ", C3.solver)
    print("C4 solver: ", C4.solver)