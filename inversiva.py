"""Animaciones del Capítulo 2: geometría inversiva clásica.

Escenas:
  InversionDefinicion        2.1
  GeneradoresPorInversion    2.1 (v) y 2.2
  CirculosYRectas            2.3
  ConformalidadInversion     2.4
  DistanciaInversiva         2.5
  CirculosDeApolonio         2.6
  CadenaDeSteiner            2.7
  EmpaquetamientoDescartes   2.8
"""

from __future__ import annotations

import numpy as np
from manim import (
    Arc,
    Arrow,
    Circle,
    Create,
    DashedLine,
    DecimalNumber,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    Line,
    MathTex,
    PI,
    Polygon,
    RIGHT,
    MovingCameraScene,
    Scene,
    TAU,
    Text,
    UP,
    UR,
    VGroup,
    VMobject,
    ValueTracker,
    always_redraw,
    config,
)

config.background_color = "#0c1016"

AZUL = "#6aabff"
AMARILLO = "#ffd15c"
VERDE = "#7ddea0"
ROJO = "#ff7b7b"
TEAL_C = "#3ddecb"
GRIS = "#aeb6c2"
ORO = "#f0c14a"


def encabezado(texto: str, size: int = 32) -> Text:
    t = Text(texto, font_size=size, color="#ffffff")
    if t.width > 13.0:
        t.scale_to_fit_width(13.0)
    t.to_edge(UP, buff=0.28)
    return t


def pie(texto: str, size: int = 24) -> Text:
    t = Text(texto, font_size=size, color=GRIS)
    if t.width > 13.2:
        t.scale_to_fit_width(13.2)
    t.to_edge(DOWN, buff=0.28)
    return t


def v3(x, y=0.0) -> np.ndarray:
    return np.array([float(x), float(y), 0.0])


def xy(p) -> np.ndarray:
    a = np.asarray(p, dtype=float).reshape(-1)
    return a[:2].copy()


def invertir(punto, centro, radio) -> np.ndarray:
    p = xy(punto)
    c = xy(centro)
    v = p - c
    return v3(*(c + (radio**2) * v / np.dot(v, v)))


def invertir_circulo(centro, radio, centro_s, radio_s):
    """Imagen de un círculo que no pasa por el centro de inversión."""
    c = xy(centro)
    a = xy(centro_s)
    d = c - a
    k = (radio_s**2) / (np.dot(d, d) - radio**2)
    return a + k * d, abs(k) * radio


def circulo_m(centro, radio, color, ancho=3.2, opacidad=0.0):
    mob = Circle(radius=float(radio), color=color, stroke_width=ancho)
    mob.move_to(v3(*xy(centro)))
    if opacidad:
        mob.set_fill(color, opacity=opacidad)
    return mob


def muestras_segmento(a, b, n=48):
    a = xy(a)
    b = xy(b)
    return [a + t * (b - a) for t in np.linspace(0, 1, n)]


def curva_de_puntos(puntos, color, ancho=4):
    mob = VMobject(color=color, stroke_width=ancho)
    pts = [v3(*xy(p)) for p in puntos]
    mob.set_points_smoothly(pts)
    return mob


# ---------------------------------------------------------------------------
# 2.1
# ---------------------------------------------------------------------------


class InversionDefinicion(Scene):
    """ι_S está en el rayo y |x-a| · |ι(x)-a| = r². Es una involución."""

    def construct(self):
        titulo = encabezado("2.1  Inversión en un círculo")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        radio = 1.85
        centro = LEFT * 1.35 + DOWN * 0.25
        circulo = circulo_m(centro, radio, AZUL, ancho=3.5, opacidad=0.05)
        o = Dot(centro, color=ORO, radius=0.06)
        et_a = MathTex("a", color=ORO).scale(0.65).next_to(o, DOWN, buff=0.1)

        dist = ValueTracker(2.9)

        def punto():
            return centro + dist.get_value() * np.array([np.cos(0.45), np.sin(0.45), 0.0])

        def imagen():
            return centro + (radio**2 / dist.get_value()) * np.array([np.cos(0.45), np.sin(0.45), 0.0])

        rayo = always_redraw(lambda: Line(centro, punto() if dist.get_value() > radio else imagen(), color=GRIS, stroke_width=2))
        p_dot = always_redraw(lambda: Dot(punto(), color=AMARILLO, radius=0.07))
        q_dot = always_redraw(lambda: Dot(imagen(), color=VERDE, radius=0.07))
        et_p = always_redraw(lambda: MathTex("x", color=AMARILLO).scale(0.6).next_to(punto(), UP, buff=0.1))
        et_q = always_redraw(lambda: MathTex(r"\iota(x)", color=VERDE).scale(0.6).next_to(imagen(), DOWN, buff=0.1))

        d_num = DecimalNumber(dist.get_value(), num_decimal_places=2, font_size=28, color=AMARILLO)
        i_num = DecimalNumber(radio**2 / dist.get_value(), num_decimal_places=2, font_size=28, color=VERDE)
        prod = DecimalNumber(radio**2, num_decimal_places=2, font_size=28, color=ORO)
        d_num.add_updater(lambda m: m.set_value(dist.get_value()))
        i_num.add_updater(lambda m: m.set_value(radio**2 / dist.get_value()))

        fila = VGroup(
            MathTex(r"|x-a|=", color=AMARILLO).scale(0.7),
            d_num,
        ).arrange(RIGHT, buff=0.1)
        fila2 = VGroup(
            MathTex(r"|\iota(x)-a|=", color=VERDE).scale(0.7),
            i_num,
        ).arrange(RIGHT, buff=0.1)
        fila3 = VGroup(
            MathTex(r"|x-a|\,|\iota(x)-a|=", color=ORO).scale(0.7),
            prod,
        ).arrange(RIGHT, buff=0.1)
        panel = VGroup(fila, fila2, fila3).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        panel.to_corner(RIGHT + UP, buff=0.45).shift(DOWN * 0.35)
        for num, ancla in (
            (d_num, d_num.get_center().copy()),
            (i_num, i_num.get_center().copy()),
            (prod, prod.get_center().copy()),
        ):
            num.add_updater(lambda m, p=ancla: m.move_to(p))

        formula = MathTex(
            r"\iota(x)=a+\frac{r^{2}}{|x-a|^{2}}(x-a)"
        ).scale(0.62)
        formula.next_to(panel, DOWN, buff=0.35).align_to(panel, LEFT)

        self.play(Create(circulo), FadeIn(o), FadeIn(et_a), FadeIn(panel), FadeIn(formula))
        self.add(rayo, p_dot, q_dot, et_p, et_q)
        self.wait(0.3)
        self.play(dist.animate.set_value(0.72), run_time=5.5, rate_func=lambda t: t)
        nota = pie("Dentro y fuera se intercambian. Sobre el círculo, el punto queda fijo.")
        self.play(FadeIn(nota))
        self.wait(0.6)
        # Involución: la imagen del imagen es el punto.
        self.play(dist.animate.set_value(2.15), run_time=1.6)
        nota2 = pie("Hacerlo dos veces devuelve el mismo punto: la inversión es una involución.")
        self.play(FadeOut(nota), FadeIn(nota2))
        self.wait(1.6)


class GeneradoresPorInversion(Scene):
    """Dos inversiones concéntricas dan una homotecia; dos reflexiones, una rotación."""

    def construct(self):
        titulo = encabezado("2.2  El grupo sale de componer inversiones")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        centro = LEFT * 3.15 + DOWN * 0.35
        r1, r2 = 1.15, 1.62
        c1 = circulo_m(centro, r1, AZUL, 2.4)
        c2 = circulo_m(centro, r2, TEAL_C, 2.4)
        self.play(FadeIn(c1), FadeIn(c2))

        verts = [v3(0.42, 0.18), v3(0.95, 0.22), v3(0.58, 0.78)]
        tri = VMobject(color=AMARILLO, stroke_width=4)
        tri.set_points_as_corners([centro + p for p in verts] + [centro + verts[0]])
        self.play(FadeIn(tri))

        def ida_y_vuelta(punto_local, radio):
            # punto_local está medido desde `centro`, que aquí es el centro de inversión.
            v = xy(punto_local)
            return (radio**2) * v / np.dot(v, v)

        arco_pts = []
        for a, b in zip(verts, verts[1:] + verts[:1]):
            for t in np.linspace(0, 1, 20, endpoint=False):
                q = a + t * (b - a)
                arco_pts.append(centro + v3(*ida_y_vuelta(q, r1)))
        arco_pts.append(arco_pts[0])
        una = VMobject(color=ROJO, stroke_width=3.5)
        una.set_points_smoothly(arco_pts[:-1])
        self.play(FadeIn(una), run_time=1.2)

        factor = (r2 / r1) ** 2
        hom = VMobject(color=VERDE, stroke_width=4)
        hom.set_points_as_corners([centro + factor * p for p in verts] + [centro + factor * verts[0]])
        self.play(FadeIn(hom), run_time=1.2)
        et = Text("dos inversiones concéntricas\n= homotecia", font_size=22, color=VERDE)
        et.next_to(c2, DOWN, buff=0.2)
        self.play(FadeIn(et))
        self.wait(0.8)

        # Reflexiones: rotación por el doble del ángulo.
        origen = RIGHT * 2.5 + DOWN * 0.15
        ang = 0.55
        linea1 = Line(origen + LEFT * 1.5, origen + RIGHT * 1.7, color=AZUL, stroke_width=3)
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        linea2 = Line(origen - 1.7 * d, origen + 1.7 * d, color=TEAL_C, stroke_width=3)
        bandera = Polygon(
            origen + v3(0.85, 0.12),
            origen + v3(1.25, 0.12),
            origen + v3(1.25, 0.42),
            color=AMARILLO,
            fill_color=AMARILLO,
            fill_opacity=0.9,
            stroke_width=1,
        )
        self.play(FadeIn(linea1), FadeIn(linea2), FadeIn(bandera))
        self.play(bandera.animate.apply_matrix([[1, 0], [0, -1]], about_point=origen), run_time=1.1)
        # Reflexión en la recta de ángulo `ang`: conjugada de la reflexión en el eje.
        rot = np.array(
            [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
        )
        reflejo_eje = np.array([[1.0, 0.0], [0.0, -1.0]])
        matriz = rot @ reflejo_eje @ rot.T
        self.play(bandera.animate.apply_matrix(matriz, about_point=origen), run_time=1.2)
        et2 = Text("dos reflexiones\n= rotación del doble", font_size=22, color=ORO)
        et2.next_to(linea1, DOWN, buff=0.25)
        self.play(FadeIn(et2))
        nota = pie("Por eso el grupo de Möbius contiene homotecias, rotaciones y traslaciones.")
        self.play(FadeIn(nota))
        self.wait(1.8)


# ---------------------------------------------------------------------------
# 2.3
# ---------------------------------------------------------------------------


class CirculosYRectas(Scene):
    """La inversión manda círculos generalizados en círculos generalizados."""

    def construct(self):
        titulo = encabezado("2.3  Círculos y rectas son la misma familia")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        escala = 1.7
        despl = DOWN * 0.3

        def D(p):
            p = xy(p)
            return despl + escala * v3(p[0], p[1])

        unidad = circulo_m(D(v3(0, 0)), escala * 1.0, AZUL, 3.2, 0.05)
        et = MathTex(r"S", color=AZUL).scale(0.6).next_to(unidad, UP, buff=0.08)
        self.play(FadeIn(unidad), FadeIn(et))

        # 1. Círculo por el origen → recta.
        c = np.array([1.15, 0.35])
        original = circulo_m(D(c), escala * np.linalg.norm(c), AMARILLO, 3.5)
        u = c / np.linalg.norm(c)
        p = c / (2 * np.dot(c, c))
        tang = np.array([-u[1], u[0]])
        recta = Line(D(p - 2.6 * tang), D(p + 2.6 * tang), color=VERDE, stroke_width=3.5)
        ley = Text("pasa por el centro  →  recta", font_size=24, color=VERDE)
        ley.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(original), FadeIn(ley))
        self.play(Create(recta), run_time=1.2)
        self.wait(0.7)

        # 2. Círculo que no pasa por el origen → círculo.
        self.play(FadeOut(original), FadeOut(recta), FadeOut(ley))
        c2 = np.array([1.55, -0.55])
        r2 = 0.48
        img_c, img_r = invertir_circulo(c2, r2, np.zeros(2), 1.0)
        src = circulo_m(D(c2), escala * r2, AMARILLO, 3.5)
        dst = circulo_m(D(img_c), escala * img_r, VERDE, 3.5)
        ley = Text("no pasa por el centro  →  otro círculo", font_size=24, color=VERDE)
        ley.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(src), FadeIn(ley))
        self.play(Create(dst), run_time=1.2)
        self.wait(0.7)

        # 3. Recta por el origen → ella misma. Un punto y su imagen siguen sobre ella.
        self.play(FadeOut(src), FadeOut(dst), FadeOut(ley))
        eje = Line(D(v3(-2.3, 0)), D(v3(2.3, 0)), color=AMARILLO, stroke_width=3)
        t = ValueTracker(1.8)
        punto = always_redraw(lambda: Dot(D(v3(t.get_value(), 0)), color=AMARILLO, radius=0.06))
        imagen = always_redraw(lambda: Dot(D(v3(1.0 / t.get_value(), 0)), color=VERDE, radius=0.06))
        ley = Text("recta por el centro  →  ella misma", font_size=24, color=VERDE)
        ley.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(eje), FadeIn(ley))
        self.add(punto, imagen)
        self.play(t.animate.set_value(0.55), run_time=2.4)
        self.wait(1.2)


# ---------------------------------------------------------------------------
# 2.4
# ---------------------------------------------------------------------------


class ConformalidadInversion(Scene):
    """El ángulo se conserva, la orientación se invierte, y lo ortogonal queda fijo."""

    def construct(self):
        titulo = encabezado("2.4  La inversión conserva el ángulo e invierte el sentido")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        escala = 1.65
        despl = LEFT * 0.2 + DOWN * 0.35

        def D(p):
            p = xy(p)
            return despl + escala * v3(p[0], p[1])

        unidad = circulo_m(D(v3(0, 0)), escala, AZUL, 3.0, 0.04)
        self.play(FadeIn(unidad))

        p = np.array([1.4, 0.45])
        imagen = p / np.dot(p, p)
        ang0 = np.arctan2(p[1], p[0]) - 0.5
        delta = 1.15

        def flechas(punto, direcciones, colores):
            grupo = VGroup()
            for d, col in zip(direcciones, colores):
                d = d / np.linalg.norm(d)
                grupo.add(
                    Arrow(
                        D(punto),
                        D(punto + 0.48 * d),
                        buff=0,
                        color=col,
                        stroke_width=4,
                        max_tip_length_to_length_ratio=0.28,
                    )
                )
            return grupo

        d1 = np.array([np.cos(ang0), np.sin(ang0)])
        d2 = np.array([np.cos(ang0 + delta), np.sin(ang0 + delta)])
        src = flechas(p, [d1, d2], [AMARILLO, TEAL_C])
        # Dι es un múltiplo de la reflexión en la recta perpendicular a p.
        u = p / np.linalg.norm(p)

        def reflejar(v):
            return v - 2 * np.dot(v, u) * u

        dst = flechas(imagen, [reflejar(d1), reflejar(d2)], [AMARILLO, TEAL_C])
        arco1 = Arc(
            radius=0.28 * escala,
            start_angle=ang0,
            angle=delta,
            arc_center=D(p),
            color=ORO,
            stroke_width=3,
        )
        # El arco imagen recorre las direcciones reflejadas, en sentido contrario.
        a_img = np.arctan2(reflejar(d1)[1], reflejar(d1)[0])
        a_img2 = np.arctan2(reflejar(d2)[1], reflejar(d2)[0])
        giro = (a_img2 - a_img + PI) % TAU - PI
        arco2 = Arc(
            radius=0.28 * escala,
            start_angle=a_img,
            angle=giro,
            arc_center=D(imagen),
            color=ORO,
            stroke_width=3,
        )
        self.play(FadeIn(src), FadeIn(arco1))
        self.play(FadeIn(dst), FadeIn(arco2))
        et = MathTex(rf"{np.degrees(delta):.0f}^\circ", color=ORO).scale(0.55)
        et.next_to(arco1, UP, buff=0.08)
        et2 = MathTex(rf"{np.degrees(abs(giro)):.0f}^\circ", color=ORO).scale(0.55)
        et2.next_to(arco2, DOWN, buff=0.08)
        self.play(FadeIn(et), FadeIn(et2))
        nota = pie("Mismo ángulo. El sentido del arco se da vuelta: la inversión es anticonforme.")
        self.play(FadeIn(nota))
        self.wait(1.3)

        self.play(FadeOut(src), FadeOut(dst), FadeOut(arco1), FadeOut(arco2), FadeOut(et), FadeOut(et2), FadeOut(nota))
        # Círculo ortogonal a la unidad: |c|² = r² + 1.
        c = np.array([1.5, 0.2])
        r = float(np.sqrt(np.dot(c, c) - 1))
        ort = circulo_m(D(c), escala * r, VERDE, 3.5)
        ley = Text("Si corta a S en ángulo recto, la inversión lo deja en su sitio.", font_size=24, color=GRIS)
        ley.to_edge(DOWN, buff=0.35)
        self.play(Create(ort), FadeIn(ley))

        phi = ValueTracker(0.4)

        def sobre():
            return c + r * np.array([np.cos(phi.get_value()), np.sin(phi.get_value())])

        punto = always_redraw(lambda: Dot(D(sobre()), color=AMARILLO, radius=0.06))
        otro = always_redraw(lambda: Dot(D(sobre() / np.dot(sobre(), sobre())), color=ORO, radius=0.06))
        self.add(punto, otro)
        self.play(phi.animate.set_value(phi.get_value() + 3.4), run_time=4.5, rate_func=lambda t: t)
        self.wait(0.8)


# ---------------------------------------------------------------------------
# 2.5
# ---------------------------------------------------------------------------


class DistanciaInversiva(Scene):
    """δ = (r1² + r2² - d²) / (2 r1 r2). Vale cos φ cuando los círculos se cortan."""

    def construct(self):
        titulo = encabezado("2.5  La distancia inversiva, el primer invariante métrico")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        r = 1.15
        separacion = ValueTracker(1.15)
        base = DOWN * 0.35

        def centros():
            d = separacion.get_value()
            return base + LEFT * d / 2, base + RIGHT * d / 2

        c1 = always_redraw(lambda: circulo_m(centros()[0], r, AZUL, 3.4))
        c2 = always_redraw(lambda: circulo_m(centros()[1], r, TEAL_C, 3.4))

        def delta():
            d = separacion.get_value()
            return (2 * r * r - d * d) / (2 * r * r)

        num = DecimalNumber(delta(), num_decimal_places=2, include_sign=True, font_size=36, color=ORO)
        num.add_updater(lambda m: m.set_value(delta()))
        etiqueta = MathTex(r"\delta=", color=ORO).scale(0.9)
        fila = VGroup(etiqueta, num).arrange(RIGHT, buff=0.12)
        fila.to_corner(UR, buff=0.5).shift(DOWN * 0.2)
        ancla = num.get_center().copy()
        num.add_updater(lambda m, p=ancla: m.move_to(p))
        formula = MathTex(
            r"\delta=\frac{r_1^{2}+r_2^{2}-d^{2}}{2 r_1 r_2}"
        ).scale(0.62)
        formula.next_to(fila, DOWN, buff=0.25)

        self.play(FadeIn(c1), FadeIn(c2), FadeIn(fila), FadeIn(formula))
        estado = Text("se cortan:  δ = cos φ", font_size=26, color=GRIS)
        estado.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(estado))
        self.wait(0.6)
        self.play(FadeOut(estado), separacion.animate.set_value(r * np.sqrt(2)), run_time=2.0)
        ort = Text("ortogonales:  δ = 0", font_size=26, color=VERDE).move_to(estado)
        self.play(FadeIn(ort))
        self.wait(0.8)
        self.play(FadeOut(ort), separacion.animate.set_value(2 * r), run_time=1.8)
        tang = Text("tangentes por fuera:  δ = −1", font_size=26, color=AMARILLO).move_to(estado)
        self.play(FadeIn(tang))
        self.wait(0.8)
        self.play(FadeOut(tang), separacion.animate.set_value(3.15), run_time=1.8)
        lejos = Text("separados:  |δ| > 1,   distancia = arccosh|δ|", font_size=26, color=ORO).move_to(estado)
        if lejos.width > 12.5:
            lejos.scale_to_fit_width(12.5)
        self.play(FadeIn(lejos))
        self.wait(1.6)


# ---------------------------------------------------------------------------
# 2.6
# ---------------------------------------------------------------------------


class CirculosDeApolonio(Scene):
    """El lugar |PA|/|PB| = k, y la familia ortogonal que pasa por A y B."""

    def construct(self):
        titulo = encabezado("2.6  Círculos de Apolonio y el haz ortogonal")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        A = np.array([-1.0, 0.0])
        B = np.array([1.0, 0.0])
        escala = 1.15
        origen_izq = LEFT * 3.35 + DOWN * 0.3

        def D(p):
            return origen_izq + escala * v3(p[0], p[1])

        familia = VGroup()
        for k, col in ((0.5, AZUL), (0.72, AZUL), (1.45, TEAL_C), (2.0, TEAL_C)):
            centro = (k**2 * B - A) / (k**2 - 1.0)
            radio = k * np.linalg.norm(B - A) / abs(k**2 - 1.0)
            familia.add(circulo_m(D(centro), escala * radio, col, 2.2))
        mediatriz = Line(D(v3(0, -2.15)), D(v3(0, 2.15)), color=ORO, stroke_width=2.4)
        haz = VGroup()
        for t in (1.15, 2.0, -1.5):
            centro = np.array([0.0, t])
            radio = float(np.hypot(1.0, t))
            haz.add(circulo_m(D(centro), escala * radio, GRIS, 1.8))
        a_dot = Dot(D(A), color=AMARILLO, radius=0.06)
        b_dot = Dot(D(B), color=AMARILLO, radius=0.06)
        et_a = MathTex("A", color=AMARILLO).scale(0.55).next_to(a_dot, DOWN, buff=0.08)
        et_b = MathTex("B", color=AMARILLO).scale(0.55).next_to(b_dot, DOWN, buff=0.08)

        self.play(FadeIn(haz), FadeIn(mediatriz))
        self.play(FadeIn(familia), FadeIn(a_dot), FadeIn(b_dot), FadeIn(et_a), FadeIn(et_b))

        # La imagen por w = (z-A)/(z-B): concéntricas y rayos.
        origen_der = RIGHT * 3.15 + DOWN * 0.3
        rayos = VGroup(
            *[
                Line(
                    origen_der,
                    origen_der + 2.15 * np.array([np.cos(a), np.sin(a), 0]),
                    color=GRIS,
                    stroke_width=1.8,
                )
                for a in np.linspace(0, TAU, 8, endpoint=False)
            ]
        )
        conc = VGroup(
            *[
                Circle(radius=rad, color=col, stroke_width=2.4).move_to(origen_der)
                for rad, col in ((0.55, AZUL), (0.85, AZUL), (1.35, TEAL_C), (1.8, TEAL_C))
            ]
        )
        cero = Dot(origen_der, color=AMARILLO, radius=0.06)
        et0 = MathTex("0", color=AMARILLO).scale(0.55).next_to(cero, DOWN, buff=0.08)
        et_inf = MathTex(r"\infty", color=AMARILLO).scale(0.6)
        et_inf.next_to(conc[-1], UP, buff=0.1)
        self.play(FadeIn(rayos), FadeIn(conc), FadeIn(cero), FadeIn(et0), FadeIn(et_inf))
        nota = pie("A y B son inversos en cada círculo. Una Möbius los manda a 0 e infinito.")
        self.play(FadeIn(nota))
        self.wait(2.0)


# ---------------------------------------------------------------------------
# 2.7
# ---------------------------------------------------------------------------


class CadenaDeSteiner(Scene):
    """Si la cadena cierra para un círculo inicial, cierra para cualquiera."""

    def construct(self):
        titulo = encabezado("2.7  El porisma de Steiner")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        n = 6
        r_int = 0.55
        R_ext = 1.65
        rho = (R_ext - r_int) / 2
        m = (R_ext + r_int) / 2
        fase = ValueTracker(0.15)

        # Inversión que descentra la cadena, sin reventar los radios.
        a_inv = np.array([2.35, 0.0])
        R_inv = 1.15

        def cadena(angulo):
            return [
                (
                    m * np.array([np.cos(angulo + TAU * j / n), np.sin(angulo + TAU * j / n)]),
                    rho,
                )
                for j in range(n)
            ]

        def panel_izq():
            origen = LEFT * 3.45 + DOWN * 0.25
            s = 1.05
            grupo = VGroup(
                circulo_m(origen, s * r_int, AZUL, 3),
                circulo_m(origen, s * R_ext, AZUL, 3),
            )
            for c, r in cadena(fase.get_value()):
                grupo.add(circulo_m(origen + s * v3(*c), s * r, AMARILLO, 2.6))
            return grupo

        def panel_der():
            circulos = [(np.zeros(2), r_int), (np.zeros(2), R_ext)] + cadena(fase.get_value())
            imgs = [invertir_circulo(c, r, a_inv, R_inv) for c, r in circulos]
            # Centrar el racimo.
            centros = np.stack([c for c, _ in imgs])
            medio = centros.mean(axis=0)
            origen = RIGHT * 3.15 + DOWN * 0.25
            s = 1.55
            colores = [AZUL, AZUL] + [AMARILLO] * n
            anchos = [3, 3] + [2.6] * n
            grupo = VGroup()
            for (c, r), col, an in zip(imgs, colores, anchos):
                grupo.add(circulo_m(origen + s * v3(*(c - medio)), s * r, col, an))
            return grupo

        izq = always_redraw(panel_izq)
        der = always_redraw(panel_der)
        et_i = Text("concéntricos", font_size=22, color=GRIS).move_to(LEFT * 3.45 + UP * 2.15)
        et_d = Text("su inversa", font_size=22, color=GRIS).move_to(RIGHT * 3.15 + UP * 2.15)
        self.play(FadeIn(izq), FadeIn(der), FadeIn(et_i), FadeIn(et_d))
        nota = pie("Se cierra con 6. Al deslizar el primero, sigue cerrando.")
        self.play(FadeIn(nota))
        self.play(fase.animate.set_value(fase.get_value() + TAU / n), run_time=6, rate_func=lambda t: t)
        self.wait(1.0)


# ---------------------------------------------------------------------------
# 2.8
# ---------------------------------------------------------------------------


class EmpaquetamientoDescartes(MovingCameraScene):
    """Tres círculos tangentes y las dos soluciones de Descartes, más un paso del empaquetamiento."""

    def construct(self):
        titulo = encabezado("2.8  Descartes: la curvatura del cuarto círculo")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        escala = 1.05
        despl = LEFT * 1.55 + DOWN * 0.15
        lado = 2.0
        dist_centro = lado / np.sqrt(3.0)
        centros = []
        for k in range(3):
            ang = PI / 2 + TAU * k / 3
            centros.append(dist_centro * np.array([np.cos(ang), np.sin(ang)]))
        radio = lado / 2

        def P(z):
            return despl + escala * v3(z[0], z[1])

        tres = VGroup(
            *[circulo_m(P(c), escala * radio, AZUL, 3.2, 0.08) for c in centros]
        )
        self.play(FadeIn(tres))

        k = 1.0 / radio
        k_in = 3 * k + 2 * np.sqrt(3 * k * k)
        k_out = 3 * k - 2 * np.sqrt(3 * k * k)
        interno = circulo_m(P(np.zeros(2)), escala / k_in, AMARILLO, 3.2, 0.5)
        externo = circulo_m(P(np.zeros(2)), escala / abs(k_out), TEAL_C, 2.6)
        self.play(Create(interno))
        self.play(Create(externo))

        formula = VGroup(
            MathTex(r"k_4=k_1+k_2+k_3").scale(0.62),
            MathTex(r"\pm\,2\sqrt{k_1k_2+k_1k_3+k_2k_3}").scale(0.62),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        formula.to_edge(RIGHT, buff=0.4).shift(UP * 1.45)
        nums = MathTex(
            rf"k_1=k_2=k_3={k:.2f}",
            rf"k_+={k_in:.2f}",
            rf"k_-={k_out:.2f}",
        ).scale(0.58)
        nums.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        nums.next_to(formula, DOWN, buff=0.28).align_to(formula, LEFT)
        self.play(FadeIn(formula), FadeIn(nums))

        # Un círculo más en cada hueco entre dos de los grandes y el interno.
        z = [complex(c[0], c[1]) for c in centros]
        k3 = k_in
        z3 = 0j
        cusps = VGroup()
        for i, j in ((0, 1), (1, 2), (2, 0)):
            centro, radio_c = _soddy(k, z[i], k, z[j], k3, z3)
            cusps.add(circulo_m(P((centro.real, centro.imag)), escala * radio_c, VERDE, 2.6, 0.45))
        self.play(FadeIn(cusps), run_time=1.2)
        nota = pie("La curvatura negativa es el círculo grande. Luego se entra en un hueco.")
        self.play(FadeIn(nota))
        self.wait(0.6)
        self.play(
            FadeOut(titulo),
            FadeOut(formula),
            FadeOut(nums),
            FadeOut(nota),
            self.camera.frame.animate.scale(0.16).move_to(cusps[0].get_center()),
            run_time=2.4,
        )
        self.wait(1.6)


def _soddy(k1, z1, k2, z2, k3, z3):
    """La solución pequeña del teorema de Descartes, como número complejo."""
    base = k1 * k2 + k1 * k3 + k2 * k3
    k4 = k1 + k2 + k3 + 2 * np.sqrt(base)
    radical = np.sqrt(k1 * k2 * z1 * z2 + k1 * k3 * z1 * z3 + k2 * k3 * z2 * z3 + 0j)
    candidatos = []
    for signo_r in (1, -1):
        z4 = (k1 * z1 + k2 * z2 + k3 * z3 + signo_r * 2 * radical) / k4
        candidatos.append(z4)
    # El del hueco es el más cercano al baricentro de los tres centros.
    meta = (z1 + z2 + z3) / 3
    centro = min(candidatos, key=lambda z: abs(z - meta))
    return centro, 1.0 / k4
