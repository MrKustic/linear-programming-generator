import math
import numpy as np
from numpy import ndarray, dtype
from scipy.spatial import ConvexHull
from typing import Tuple, Any, List
from pylatex import Document, Section, Command, Math, NewPage, HugeText, Center, Tabular, NewLine, \
    FlushRight, TikZ, TikZCoordinate, TikZOptions, TikZDraw, TikZNode, FlushLeft, TikZPathList
from fractions import Fraction
from decimal import Decimal
from scipy.optimize import linprog


class PolygonData:
    def __init__(self, points: ndarray, opt_point: ndarray, func: ndarray, max_value: int = 0):
        self.polygon = points
        self.opt_point = opt_point
        self.func = func


class Statement:
    def __init__(self, ineqs, b, func, zero_is_allowed):
        self.ineqs = ineqs
        self.b = b
        self.func = func
        self.zero_is_allowed = zero_is_allowed


class StatementGenerator:
    def __init__(self, n: int, ineq_max_value: int, func_max_value: int, cnt_steps: int, zero_is_allowed: bool = True):
        self.n = n
        self.ineq_max_value = ineq_max_value
        self.func_max_value = func_max_value
        self.cnt_steps = cnt_steps
        self.zero_is_allowed = zero_is_allowed

    def GenerateRandomPolygon(self) -> ndarray[Any, dtype[Any]] | list[Any]:
        max_value = self.ineq_max_value
        n = self.n
        if max_value < 0:
            raise ValueError("max_value must be non-negative")
        if n < 3:
            raise ValueError("n must be no less than 3")
        points = np.array([])
        count = 0
        m = int(1.5 ** n)
        while len(points) != n:
            new_points = np.random.randint(0, max_value + 1, size=(m, 2))
            if self.zero_is_allowed:
                minx = np.min(new_points, axis=0)
                new_points = np.subtract(new_points, np.full((m, 2), minx))
                if [0, 0] in new_points.tolist():
                    continue
                new_points = np.append(new_points, [[0, 0]], axis=0)
            else:
                if [0, 0] in new_points.tolist():
                    continue
            points = new_points[ConvexHull(new_points).vertices]
            count += 1
        return points

    def CreateFunctional(self, points: ndarray, idx_start: int = 0):
        max_value = self.func_max_value
        cnt_steps = self.cnt_steps
        if max_value <= 0:
            raise ValueError("max_value must be no less than 1")
        n = points.shape[0]
        if cnt_steps > n // 2:
            raise ValueError("the number of vertexes is not enough for such count of steps")
        idx_opt = (idx_start - cnt_steps if np.random.rand() < 0.5 else idx_start + cnt_steps) % n
        vec_prev = points[idx_opt] - points[(idx_opt - 1) % n]
        vec_nxt = points[(idx_opt + 1) % n] - points[idx_opt]
        eps = 10 ** -5
        min_tan = -math.inf if vec_prev[0] == 0 else vec_prev[1] / vec_prev[0] + eps
        max_tan = math.inf if vec_nxt[0] == 0 else vec_nxt[1] / vec_nxt[0] - eps
        if min_tan > max_tan:
            if np.random.rand() < 0.5:
                min_tan = -math.inf
            else:
                max_tan = math.inf
        for i in range(100):
            denom = np.random.randint(1, max_value + 1)
            lv = math.ceil(max(min_tan * denom, -max_value))
            rv = math.floor(min(max_tan * denom, max_value))
            if rv < lv:
                continue
            new_vec = np.array([denom, np.random.randint(lv, rv + 1)])
            break
        else:
            raise RuntimeError("too big angle, no such point")
        new_vec = np.array([-new_vec[1], new_vec[0]])
        if np.sum(new_vec * points[idx_opt]) < np.sum(new_vec * points[(idx_opt + 1) % n]):
            new_vec *= -1
        gc = np.gcd.reduce(new_vec)
        new_vec //= gc
        return idx_opt, new_vec

    def CreateStatement(self):
        n = self.n
        ineq_max_value = self.ineq_max_value
        zero_is_allowed = self.zero_is_allowed
        for try_num in range(100):
            try:
                polygon = self.GenerateRandomPolygon()
                idx_start = 0
                for j in range(n):
                    if (polygon[j] == np.array([0, 0])).all():
                        idx_start = j
                idx_opt, func = self.CreateFunctional(polygon, idx_start)
                break
            except RuntimeError:
                pass
        else:
            raise RuntimeError("func_max_value is too small, try increasing it")
        ineqs = np.array([])
        b = np.array([])
        for i in range(n):
            x1 = polygon[i]
            x2 = polygon[(i + 1) % n]  # (x2.y - x1.y) * x.x + (x1.x - x2.x) * x.y <= x2.x * x1.y - x2.y * x1.x
            line = np.array([x2[1] - x1[1], x1[0] - x2[0], -x2[0] * x1[1] + x2[1] * x1[0]])
            g = abs(np.gcd.reduce(line))
            line //= g
            if (line[0] == 0 or line[1] == 0) and line[2] == 0:
                continue
            ineqs = np.append(ineqs, [line[0:2]])
            b = np.append(b, [line[2]])
        ineqs = np.reshape(ineqs, (len(ineqs) // 2, 2))
        return Statement(ineqs, b, func, zero_is_allowed), PolygonData(polygon, polygon[idx_opt], func)


class SimplexTable:
    def __init__(self, mat: ndarray, basis: list, m: int, col: int = -1, row: int = -1):
        self.col = col
        self.row = row
        self.m = m
        self.n = len(mat) - 1
        self.k = len(mat[0]) - 1 - self.n - self.m
        values = [["0"] * (len(mat[0]) + 1) for _ in range(len(mat) + 1)]
        values[0][0] = "Базис"
        values[0][-1] = "Решение"
        values[1][0] = "z"
        for i in range(len(basis)):
            values[i + 2][0] = self.IdToLatex(basis[i])
        for i in range(m):
            values[0][i + 1] = "x_{" + str(i + 1) + "}"
        for i in range(self.n):
            values[0][i + m + 1] = "s_{" + str(i + 1) + "}"
        for i in range(self.k):
            values[0][i + self.n + self.m + 1] = "r_{" + str(i + 1) + "}"
        values[1][1:] = list(map(self.FracToLatex, mat[-1]))
        for i in range(len(mat) - 1):
            values[i + 2][1:] = list(map(self.FracToLatex, mat[i]))
        self.values = values
        if row == -1:
            self.answer = self.FracToLatex(mat[-1][-1])
            self.opt_values = ["0"] * (self.n + self.m + self.k)
            for i in range(len(mat) - 1):
                self.opt_values[basis[i]] = self.FracToLatex(mat[i][-1])
            return

        deltas = [[""] * 4 for _ in range(len(basis) + 1)]
        deltas[0] = ["Базис", self.IdToLatex(col), "Решение", "Отношение"]
        for i in range(len(basis)):
            s = r"\infty \: \text{(не подходит)}" if mat[i][col] == 0 else self.FracToLatex(mat[i][-1] / mat[i][col])
            if mat[i][col] != 0 and mat[i][-1] / mat[i][col] < 0:
                s += r"\: (< 0, \: \text{не подходит})"
            if i == row:
                s += r"\: \text{(Оптимально)}"
            deltas[i + 1] = [values[i + 2][0], self.FracToLatex(mat[i][col]), self.FracToLatex(mat[i][-1]), s]
        self.deltas = deltas

    def FracToLatex(self, frac: Fraction) -> str:
        numer, denom = frac.numerator, frac.denominator
        if denom == 1:
            return str(numer)
        return ("-" if numer < 0 else "") + r"\frac{" + str(abs(numer)) + "}{" + str(denom) + "}"

    def IdToLatex(self, id: int) -> str:
        return ("x_{" + str(id + 1) if id < self.m else "s_{" + str(
            id - self.m + 1) if id < self.n + self.m else "r_{" + str(id - self.m - self.n + 1)) + "}"


class Solution:
    def __init__(self, cnt_steps_first: int, cnt_steps_second: int, solution_first: List[SimplexTable],
                 solution_second: List[SimplexTable], k: int, answer: Fraction, first_point: ndarray, point: ndarray):
        self.cnt_steps_first = cnt_steps_first
        self.cnt_steps_second = cnt_steps_second
        self.solution_first = solution_first
        self.solution_second = solution_second
        self.k = k
        self.answer = answer
        self.point = point
        self.first_point = first_point


class Simplex:
    def __init__(self):
        self.k = 0
        self.n = self.m = 0

    def CalcOptDelta(self, a):
        id_add = np.argmin(a[-1:, :-1 - self.k])
        row = -1
        for i in range(self.n):
            if a[i][id_add] > 0:
                if row == -1 or a[i][-1] / a[i][id_add] < a[row][-1] / a[row][id_add]:
                    row = i
        if row == -1:
            raise RuntimeError("No solution")
        return id_add, row

    def Pivot(self, a, ids, id_add, row):
        a[row] /= a[row][id_add]
        for i in range(self.n + 1):
            if i == row:
                continue
            a[i] -= a[i][id_add] * a[row]
        ids[row] = id_add

    def Simplex(self, statement: Statement) -> Solution:
        to_frac = np.vectorize(lambda p: Fraction(Decimal(str(p))))
        ineqs, b, z = to_frac(statement.ineqs), to_frac(statement.b), to_frac(statement.func)
        self.n = n = len(ineqs)
        self.m = m = len(z)
        bad_rows = np.reshape(np.argwhere(b < 0), (-1))
        k = len(bad_rows)
        self.k = k
        a = np.concatenate((ineqs, to_frac(np.zeros((n, n + k)))), axis=1)
        a = np.append(a, np.reshape(b, (n, 1)), axis=1)
        for i in bad_rows:
            a[i] *= -1
        ids = [0] * n
        last_add_var = 0
        for i in range(n):
            if b[i] >= 0:
                basis_var = m + i
            else:
                basis_var = n + m + last_add_var
                last_add_var += 1
                a[i][m + i] = Fraction(-1)
            ids[i] = basis_var
            a[i][basis_var] = Fraction(1)
        find_first_sol = []
        cnt_steps_first_sol = 0
        first_point = np.zeros((m,))
        if k > 0:
            a = np.append(a, to_frac(
                np.concatenate((np.zeros((1, n + m)), np.ones((1, k)), np.zeros((1, 1))), axis=1)), axis=0)
            for i in bad_rows:
                a[-1] -= a[i]
            while np.min(a[-1:, :-1]) < 0:
                cnt_steps_first_sol += 1
                id_add, row = self.CalcOptDelta(a)
                find_first_sol.append(SimplexTable(a, ids, m, id_add, row))
                self.Pivot(a, ids, id_add, row)
            find_first_sol.append(SimplexTable(a, ids, m))
            for i in range(n):
                if ids[i] < m:
                    first_point[ids[i]] = a[i][-1]
            a = np.delete(a, -1, 0)
        z *= -1
        z = np.concatenate((z, to_frac(np.zeros(n + k + 1))))
        a = np.append(a, np.reshape(z, (1, -1)), axis=0)
        for i in range(n):
            if ids[i] < m:
                a[-1] -= a[i] * a[-1][ids[i]]
        solution = []
        cnt_steps = 0
        while np.min(a[-1:, :-1 - k]) < 0:
            cnt_steps += 1
            id_add, row = self.CalcOptDelta(a)
            solution.append(SimplexTable(a, ids, m, id_add, row))
            self.Pivot(a, ids, id_add, row)
        solution.append(SimplexTable(a, ids, m))
        result = np.zeros(m)
        for i in range(n):
            if ids[i] < m:
                result[ids[i]] = a[i][-1]
        return Solution(cnt_steps_first_sol, cnt_steps, find_first_sol, solution, k, a[-1][-1], np.array(first_point),
                        np.array(result))


class Writer:
    def __init__(self, m: int, zero_is_allowed: bool, only_latex: bool):
        self.m = m
        self.k = 0
        self.n = 0
        self.zero_is_allowed = zero_is_allowed
        self.only_latex = only_latex
        self.swapped_functionals = []

    def IdToLatex(self, id: int) -> str:
        return ("x_{" + str(id + 1) if id < self.m else "s_{" + str(
            id - self.m + 1) if id < self.n + self.m else "r_{" + str(id - self.m - self.n + 1)) + "}"

    def MonomToString(self, coef, id, first) -> Tuple[str, bool]:
        coef = int(coef)
        if coef == 0:
            return r"", False
        s = r""
        if not first and coef > 0:
            s += "+"
        if coef == -1:
            s += "-"
        if abs(coef) > 1:
            s += str(coef)
        s += self.IdToLatex(id)
        return s, True

    def InequalityToString(self, ineq, b, swap=-1, eq: int = 0) -> str:
        if swap == -1:
            swap = np.random.rand() < 0.5
        mult = -1 if swap else 1
        s = r""
        first = True
        for j in range(len(ineq)):
            curr_string, real = self.MonomToString(ineq[j] * mult, j, first)
            if real:
                first = False
            s += curr_string
        s += " & " + (r"=" if eq else r"\ge" if swap else r"\le")
        s += str(int(b * mult))
        s += r" \\"
        return s

    def CreateEmptyFile(self) -> Document:
        doc = Document(documentclass="article")
        doc.preamble.append(Command("usepackage", "fontenc", "T2A"))
        doc.preamble.append(Command("usepackage", "geometry"))
        doc.preamble.append(Command("geometry", "top=0.5in, bottom=0.5in, left=0.5in, right=0.5in"))
        doc.preamble.append(Command("usepackage", "babel", "russian"))
        doc.preamble.append(Command("usepackage", "newtxmath, newtxtext"))
        #doc.preamble.append(Command("usepackage", "fontspec"))
        #doc.preamble.append(Command("setmainfont", "Times New Roman"))
        doc.preamble.append(Command("usepackage", "amsmath"))
        return doc

    def WriteStatement(self, statements: list[Statement], path: str):
        doc = self.CreateEmptyFile()
        for var_num in range(len(statements)):
            with doc.create(Center()) as environment:
                environment.append(HugeText(f"Вариант {var_num + 1}"))
            ineqs, b, func = statements[var_num].ineqs, statements[var_num].b, statements[var_num].func
            self.n = len(ineqs)

            with doc.create(Section("№ 1", numbering=False)):
                doc.append(r"Решите следующую задачу линейного программирования: ")
                s = r"\left\{\begin{aligned}"
                for i in range(len(ineqs)):
                    ineq = ineqs[i]
                    if np.sum(np.where(ineq == 0, 0, 1)) > 1 or b[i] != 0:
                        cnt_positive = np.sum(np.where(ineq > 0, 1, 0))
                        cnt_neg = np.sum(np.where(ineq < 0, 1, 0))
                        if cnt_positive == 0:
                            s += self.InequalityToString(ineq, b[i], 1)
                        elif cnt_neg == 0:
                            s += self.InequalityToString(ineq, b[i], 0)
                        else:
                            s += self.InequalityToString(ineq, b[i])
                for i in range(len(statements[var_num].func)):
                    s += r"x_{" + str(i + 1) + "},"
                s = s[:-1] + r" & \ge 0 \\"
                s += r"\end{aligned}\right."
                doc.append(Math(data=[s], escape=False))
                cnt_positive = np.sum(np.where(func > 0, 1, 0))
                cnt_neg = np.sum(np.where(func < 0, 1, 0))
                swap = 1 if cnt_positive == 0 else 0 if cnt_neg == 0 else np.random.rand() < 0.5
                functional = self.InequalityToString(func, 0, swap)[: -9]
                functional += r" \to " + ("min" if swap else "max")
                self.swapped_functionals.append(swap)
                doc.append(Math(data=[functional], escape=False))
            doc.append(NewPage())

        if self.only_latex:
            doc.generate_tex(path)
        else:
            doc.generate_pdf(path, clean_tex=False)

    def WriteTable(self, doc: Document, table: SimplexTable) -> Document:
        with doc.create(Tabular("|c|" + "c" * (len(table.values[0]) - 2) + "|c|", row_height=1.3)) as output_table:
            output_table.add_hline()
            for i in range(len(table.values)):
                curr_row = [Math(data=[table.values[i][j]], escape=False, inline=True) for j in
                            range(len(table.values[0]))]
                if i == 0:
                    curr_row[0] = table.values[i][0]
                    curr_row[-1] = table.values[i][-1]
                output_table.add_row(curr_row)
                if i <= 1 or i == len(table.values) - 1:
                    output_table.add_hline()

        if table.col == -1:
            return doc
        for i in range(3):
            doc.append(NewLine())

        with doc.create(Tabular("|cccc|")) as output_table:
            output_table.add_hline()
            for i in range(len(table.deltas)):
                curr_row = [Math(data=[table.deltas[i][j]], escape=False, inline=True) for j in range(4)]
                if i == 0:
                    for j in [0, 2, 3]:
                        curr_row[j] = table.deltas[i][j]
                output_table.add_row(curr_row)
                if i == 0 or i == len(table.deltas) - 1:
                    output_table.add_hline()
        for i in range(3):
            doc.append(NewLine())
        return doc

    def FindIntersections(self, x_min: Tuple, x_max: Tuple, vec: ndarray, point: ndarray, k: float) -> tuple[
        TikZCoordinate, ...]:
        a, b = vec
        c = -a * point[0] - b * point[1]
        if a == 0 and b == 0:
            raise RuntimeError("invalid functional")
        if b == 0:
            return TikZCoordinate(-c / a * k, x_min[1] * k), TikZCoordinate(-c / a * k, x_max[1] * k)
        if a == 0:
            return TikZCoordinate(x_min[0] * k, -c / b * k), TikZCoordinate(x_max[0] * k, -c / b * k)
        result = []
        eps = 10 ** -7
        st = set()
        for x in [x_min[0], x_max[0]]:
            y = (-c - a * x) / b
            approx = (int(round(x * 10 ** 6)), int(round(y * 10 ** 6)))
            if x_min[1] - eps <= y <= x_max[1] + eps and not approx in st:
                result.append(TikZCoordinate(x * k, y * k))
                st.add(approx)
        for y in [x_min[1], x_max[1]]:
            x = (-c - b * y) / a
            approx = (int(round(x * 10 ** 6)), int(round(y * 10 ** 6)))
            if x_min[0] - eps <= x <= x_max[0] + eps and not approx in st:
                result.append(TikZCoordinate(x * k, y * k))
                st.add(approx)
        if len(result) != 2:
            raise RuntimeError("unexpected count of intersections")
        return tuple(result)

    def WritePlot(self, doc: Document, polygon_data: PolygonData) -> Document:
        points = polygon_data.polygon
        x1_min, x2_min = np.min(points, axis=0) * 5 / 6 - 2
        x1_min = min(0, x1_min)
        x2_min = min(0, x2_min)
        x1_max, x2_max = np.max(points, axis=0) * 1.2
        k = 10 / max(x1_max - x1_min, x2_max - x2_min)
        with doc.create(TikZ()) as pic:
            path = []
            for point in points:
                coords = TikZCoordinate(point[0] * k, point[1] * k)
                path.append(coords)
                path.append("--")
            path.append(TikZCoordinate(points[0][0] * k, points[0][1] * k))
            pic.append(TikZDraw(TikZPathList(*path), TikZOptions(fill="blue!20")))
            for point in points:
                pic.append(TikZNode(options=["above right"], text=f"$({point[0]}, {point[1]})$",
                                    at=TikZCoordinate(point[0] * k, point[1] * k)))
            pic.append(TikZDraw([TikZCoordinate(x1_min * k, 0), "--", TikZCoordinate(x1_max * k, 0)],
                                options=TikZOptions("->")))
            pic.append(TikZNode(options=["above"], at=TikZCoordinate(x1_max * k, 0), text="$x_1$"))
            pic.append(TikZDraw([TikZCoordinate(0, x2_min * k), "--", TikZCoordinate(0, x2_max * k)],
                                options=TikZOptions("->")))
            pic.append(TikZNode(options=["right"], at=TikZCoordinate(0, x2_max * k), text="$x_2$"))
            p1, p2 = self.FindIntersections((x1_min, x2_min), (x1_max, x2_max), polygon_data.func,
                                            polygon_data.opt_point, k)
            pic.append(TikZDraw([p1, "--", p2], TikZOptions("red", "dashed", "thick")))
        return doc

    def WriteSolution(self, statements: List[Statement], solutions: List[Tuple[Solution, PolygonData]],
                      path: str):
        doc = self.CreateEmptyFile()
        for var_num in range(len(solutions)):
            self.k = solutions[var_num][0].k
            with doc.create(Center()) as environment:
                environment.append(HugeText(f"Вариант {var_num + 1}"))
            solution, polygon = solutions[var_num]
            doc.append(r"Задача равносильна следующей: ")
            s = r"\left\{\begin{aligned}"
            ineqs, b, func = statements[var_num].ineqs, statements[var_num].b, statements[var_num].func
            num_last_var = 0
            for i in range(len(ineqs)):
                if b[i] >= 0:
                    ineq = np.append(ineqs[i], [[0] * (i + 1)])
                    ineq[-1] = 1
                else:
                    ineq = np.append(ineqs[i], [[0] * (self.n + num_last_var + 1)])
                    ineq[-1] = 1
                    ineq[self.m + i] = -1
                    num_last_var += 1
                s += self.InequalityToString(ineq, b[i], 0, True)
            s += ",".join(self.IdToLatex(i) for i in range(self.m + self.n + num_last_var))
            s += r" & \ge 0 \\ \end{aligned}\right."
            doc.append(Math(data=[s], escape=False))
            functional = self.InequalityToString(func, 0, 0)[: -9]
            functional += r" \to max"
            doc.append(Math(data=[functional], escape=False))
            if self.zero_is_allowed:
                doc.append("В качестве начального допустимого решения возьмем ")
                doc.append(Math(data=["(" + ", ".join(['0'] * self.m) + ")."], escape=False, inline=True))
            else:
                doc.append(
                    "Найдем начальное допустимое решение двухэтапным методом. Для этого заменим исходный функционал на следующий: ")
                doc.append(
                    Math(data=["".join("-r_{" + str(i + 1) + "}" for i in range(self.k)) + r"\to max."], escape=False))

            with doc.create(FlushLeft()):
                for i in range(len(solution.solution_first)):
                    doc.append(str(i + 1) + r"-я итерация: ")
                    doc.append(NewLine())
                    doc.append(NewLine())
                    doc = self.WriteTable(doc, solution.solution_first[i])
                if not self.zero_is_allowed:
                    doc.append(NewLine())
                    doc.append(NewLine())
                    doc.append(
                        "Таким образом, получили оптимальное значение функционала - 0 и начальное допустимое решение: ")
                    doc.append(
                        Math(data=[f"({', '.join(map(lambda p: str(int(p)), solution.first_point))})"], escape=False,
                             inline=True))
                    doc.append(NewLine())
                    doc.append(
                        "Теперь вернем исходный функционал и найдем его оптимальное значение, при этом не добавляя в базис дополнительные переменные.")
                    doc.append(NewLine())
                for i in range(len(solution.solution_second)):
                    doc.append(str(i + 1) + r"-я итерация: ")
                    doc.append(NewLine())
                    doc.append(NewLine())
                    doc = self.WriteTable(doc, solution.solution_second[i])
                doc.append(NewLine())
            swap = self.swapped_functionals[var_num]
            doc.append("Оптимальное решение найдено." + (" Необходимо домножить ответ на -1 из-за смены знака функционала." if swap else ""))
            doc.append(NewLine())
            doc = self.WritePlot(doc, polygon)
            with doc.create(FlushRight()) as environment:
                environment.append(
                    f"Ответ: {int(solution.answer * (-1 if swap else 1))}, точка - ({', '.join(map(lambda p: str(int(p)), solution.point))})")

            doc.append(NewPage())

        if self.only_latex:
            doc.generate_tex(path)
        else:
            doc.generate_pdf(path, clean_tex=False)


def Checker(statement: Statement, found_solution: Solution, cnt_steps_first: int, cnt_steps_second: int):
    if cnt_steps_first != 0 and cnt_steps_first != found_solution.solution_first:
        raise ValueError("Incorrect count of steps to find acceptable solution")
    if cnt_steps_second != found_solution.cnt_steps_second:
        raise ValueError("Incorrect count of steps")
    result = linprog(statement.func * -1, statement.ineqs, statement.b)
    if result.status != 0:
        raise RuntimeError("Incorrect statement, optimization terminated unsuccessfully")
    if np.abs(found_solution.answer + result.fun) > 10 ** -5:
        raise RuntimeError("Wrong answer")
    if np.abs(np.sum(found_solution.point - result.x)) > 10 ** -5:
        raise RuntimeError("Wrong point")


def GenerateLinearProgrammingProblems():
    np.random.seed(10)
    generator = StatementGenerator(n, ineq_max_value, func_max_value, cnt_steps_second, is_zero_allowed)
    solver = Simplex()
    statements = []
    polygons = []
    solutions = []
    for i in range(cnt_variants):
        print(f"Генерирую вариант № {i + 1}.")
        while True:
            statement, polygon = generator.CreateStatement()
            solution = solver.Simplex(statement)
            try:
                Checker(statement, solution, cnt_steps_first, cnt_steps_second)
                break
            except ValueError:
                pass
        statements.append(statement)
        polygons.append(polygon)
        solutions.append(solution)
    print("Начинаю запись в файлы.")
    writer = Writer(m, is_zero_allowed, only_latex)
    writer.WriteStatement(statements, path_to_statements)
    writer.WriteSolution(statements, list(zip(solutions, polygons)), path_to_solutions)

if __name__ == "__main__":
    n = int(input("Введите требуемое количество вершин многогранника: \n"))
    m = 2
    input_zero = input("Должно ли (0, 0) являться допустимым решением? [д/н]\n").lower()
    if input_zero != "д" and input_zero != "н":
        raise ValueError("Необходимо ввести 'д' или 'н'")
    is_zero_allowed = input_zero == 'д'
    cnt_steps_first = 0
    if not is_zero_allowed:
        cnt_steps_first = int(input("Введите требуемое количество шагов для достижения допустимого решения (0, если это не имеет значения): \n"))
    cnt_steps_second = int(input("Введите требуемое количество шагов для достижения оптимального решения: \n"))
    ineq_max_value = int(input("Введите верхнее ограничение на значения координат многогранников: \n"))
    func_max_value = int(input("Введите верхнее ограничение на абсолютные значения коэффициентов в линейном функционале: \n"))
    cnt_variants = int(input("Введите требуемое количество вариантов работ: \n"))
    path_to_statements = input("Введите название файла с условиями (без расширения): \n")
    path_to_solutions = input("Введите название файла с решениями (без расширения): \n")
    input_latex = input("Необходимо ли сразу создать pdf файлы на основе файлов tex? (для этого необходимо наличие пути до компилятора xelatex в PATH) [д/н] \n").lower()
    if input_latex != "д" and input_latex != "н":
        raise ValueError("Необходимо ввести 'д' или 'н'")
    only_latex = input_latex == 'н'
    print("Начинаю генерацию.")
    GenerateLinearProgrammingProblems()
    print("Генерация завершилась успешно.")