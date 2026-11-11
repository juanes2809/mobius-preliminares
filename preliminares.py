"""Animaciones del Capítulo 0 (Preliminares) de las notas
«Geometría de Möbius», seminario UIS, noviembre 2026.

Orden:
  0.1  EsferaYHiperplano, SemiplanosReales
  0.2  MultiplicacionCompleja, Conjugacion, Conformidad, HolomorfaYAngulo
  0.3  CorteMeridiano, FactorConforme, MetricaCordal, Estereografica3D
  0.4  OrbitaEstabilizador, EstabilizadorYDimension
  0.6  ProgramaErlangen
"""

from __future__ import annotations

import numpy as np
from manim import (

    BLUE,
    BLUE_B,
    BLUE_D,
    DEGREES,
    DOWN,
    GOLD,
    GREEN,
    LEFT,
    ORIGIN,
    OUT,
    PI,
    RED,
    RIGHT,
    TAU,
    TEAL,
    UL,
    UP,
    UR,
    WHITE,
    YELLOW,
    Angle,
    Arc,
    Arrow,
    Axes,
    Circle,
    Create,
    DashedLine,
    DecimalNumber,
    Dot,
    Dot3D,
    FadeIn,
    FadeOut,
    Line,
    Line3D,
    MathTex,
    NumberPlane,
    ParametricFunction,
    Polygon,
    RegularPolygon,
    Rotate,
    RoundedRectangle,
    Scene,
    Sphere,
    Text,
    ThreeDScene,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    ValueTracker,
    always_redraw,
    config,
)

config.background_color = "#0c1016"

# Paleta común
AZUL = "#6aabff"
AMARILLO = "#ffd15c"
VERDE = "#7ddea0"
ROJO = "#ff7b7b"
TEAL_C = "#3ddecb"
GRIS = "#aeb6c2"
ORO = "#f0c14a"


def encabezado(texto: str, size: int = 32) -> Text:
    t = Text(texto, font_size=size, color=WHITE)
    if t.width > 13.0:
        t.scale_to_fit_width(13.0)
    t.to_edge(UP, buff=0.28)
    return t


def pie(texto: str, size: int = 24) -> Text:
    t = Text(texto, font_size=size, color=GRIS)
    t.to_edge(DOWN, buff=0.28)
    if t.width > 13.2:
        t.scale_to_fit_width(13.2)
    return t


def angulo_entre(v, w) -> float:
    v = np.array(v, dtype=float)
    w = np.array(w, dtype=float)
    c = float(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def esquina_recta(origen, d1, d2, size=0.16, color=ORO):
    a = np.array(d1, dtype=float)
    b = np.array(d2, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    p1 = origen + size * a
    p2 = origen + size * a + size * b
    p3 = origen + size * b
    return VGroup(
        Line(p1, p2, color=color, stroke_width=2.6),
        Line(p2, p3, color=color, stroke_width=2.6),
    )


# ---------------------------------------------------------------------------
# 0.1
# ---------------------------------------------------------------------------


class EsferaYHiperplano(Scene):
    """S(a, r) y un hiperplano {x : x·u = c} en R^2."""

    def construct(self):
        titulo = encabezado("0.1  Esferas e hiperplanos")
        self.play(FadeIn(titulo, shift=DOWN * 0.15))

        ejes = Axes(
            x_range=[-1, 6, 1],
            y_range=[-2.2, 4.2, 1],
            x_length=8.4,
            y_length=5.4,
            tips=True,
            axis_config={"stroke_color": GRIS, "stroke_width": 2, "include_ticks": False},
        )
        ejes.shift(DOWN * 0.25)
        etiqueta_rn = MathTex(r"\mathbb{R}^2", color=GRIS).scale(0.7)
        etiqueta_rn.next_to(ejes.x_axis.get_end(), DOWN, buff=0.12)

        centro_math = np.array([2.0, 1.15])
        radio = 1.25
        centro = ejes.c2p(*centro_math)
        unidad = np.linalg.norm(ejes.c2p(1, 0) - ejes.c2p(0, 0))
        circulo = Circle(radius=radio * unidad, color=AZUL, stroke_width=3.5).move_to(centro)
        punto_a = Dot(centro, color=AMARILLO, radius=0.06)
        etiqueta_a = MathTex("a", color=AMARILLO).scale(0.7).next_to(punto_a, DOWN, buff=0.12)

        borde = ejes.c2p(centro_math[0] + radio, centro_math[1])
        radio_seg = Line(centro, borde, color=AMARILLO, stroke_width=3)
        etiqueta_r = MathTex("r", color=AMARILLO).scale(0.65)
        etiqueta_r.next_to(radio_seg.get_center(), UP, buff=0.1)

        formula_esfera = MathTex(
            r"S(a,r)=\{x:\ |x-a|=r\}", color=AZUL
        ).scale(0.78)
        formula_esfera.to_corner(UR, buff=0.45).shift(DOWN * 0.55)

        # Hiperplano x + y = 4.2, normal u = (1, 1)/sqrt(2)
        p0 = ejes.c2p(0.4, 3.8)
        p1 = ejes.c2p(5.2, -1.0)
        recta = Line(p0, p1, color=TEAL_C, stroke_width=3.5)
        # Punto de apoyo de la normal: (2.2, 2.0), porque 2.2+2.0=4.2
        apoyo = ejes.c2p(2.2, 2.0)
        normal_dir = np.array([1.0, 1.0, 0.0])
        normal_dir = normal_dir / np.linalg.norm(normal_dir)
        normal_fin = ejes.c2p(2.2, 2.0) + normal_dir * 0.95
        normal = Arrow(
            apoyo, normal_fin, buff=0, color=TEAL_C, stroke_width=4,
            max_tip_length_to_length_ratio=0.28,
        )
        etiqueta_u = MathTex("u", color=TEAL_C).scale(0.7).next_to(normal.get_end(), UP, buff=0.08)
        formula_hip = MathTex(
            r"\{x:\ x\cdot u = c\}", color=TEAL_C
        ).scale(0.78)
        formula_hip.next_to(formula_esfera, DOWN, buff=0.28, aligned_edge=LEFT)

        self.play(FadeIn(ejes), FadeIn(etiqueta_rn))
        self.play(Create(circulo))
        self.play(FadeIn(punto_a), FadeIn(etiqueta_a), FadeIn(radio_seg), FadeIn(etiqueta_r))
        self.play(FadeIn(formula_esfera, shift=LEFT * 0.2))
        self.wait(0.4)
        self.play(Create(recta))
        self.play(FadeIn(normal), FadeIn(etiqueta_u), FadeIn(formula_hip, shift=LEFT * 0.2))
        nota = pie("En R^n la esfera es |x − a| = r; el hiperplano es x · u = c.")
        self.play(FadeIn(nota))
        self.wait(1.6)


class SemiplanosReales(Scene):
    """z ↦ −z está en PGL(2,R) y no en PSL(2,R): intercambia semiplanos."""

    def construct(self):
        titulo = encabezado("0.1  PSL(2, R) tiene índice 2 en PGL(2, R)")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        ejes = Axes(
            x_range=[-3.2, 3.2, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=7.2,
            y_length=5.6,
            tips=False,
            axis_config={"stroke_color": GRIS, "stroke_width": 2, "include_ticks": False},
        )
        ejes.shift(LEFT * 1.7 + DOWN * 0.15)
        origen = ejes.c2p(0, 0)
        unidad = np.linalg.norm(ejes.c2p(1, 0) - ejes.c2p(0, 0))

        superior = Polygon(
            ejes.c2p(-3.05, 0),
            ejes.c2p(3.05, 0),
            ejes.c2p(3.05, 2.35),
            ejes.c2p(-3.05, 2.35),
            stroke_width=0,
            fill_color=AZUL,
            fill_opacity=0.38,
        )
        # Triángulo pequeño alrededor de i, para ver la orientación.
        i_pt = ejes.c2p(0, 1)
        tri = Polygon(
            ejes.c2p(-0.28, 0.72),
            ejes.c2p(0.42, 0.78),
            ejes.c2p(-0.05, 1.45),
            color=AMARILLO,
            fill_color=AMARILLO,
            fill_opacity=0.9,
            stroke_width=1.5,
        )
        punto = Dot(i_pt, color=WHITE, radius=0.055)
        et_i = MathTex("i", color=WHITE).scale(0.65).next_to(punto, LEFT, buff=0.12)
        et_h = MathTex(r"\mathbb{H}", color=AZUL).scale(0.8)
        et_h.move_to(ejes.c2p(-2.15, 1.7))

        matriz = MathTex(
            r"z\mapsto -z",
            r"\qquad",
            r"\begin{pmatrix}-1&0\\0&1\end{pmatrix}",
        ).scale(0.72)
        matriz.to_edge(RIGHT, buff=0.4).shift(UP * 1.15)
        det = MathTex(r"\det=-1", color=ROJO).scale(0.8)
        det.next_to(matriz, DOWN, buff=0.25)

        grupo = VGroup(superior, tri, punto)
        self.play(FadeIn(ejes), FadeIn(superior), FadeIn(tri), FadeIn(punto), FadeIn(et_i), FadeIn(et_h))
        self.wait(0.4)
        self.play(FadeIn(matriz), FadeIn(det))
        self.wait(0.3)

        et_i.add_updater(lambda m: m.next_to(punto, _lado_exterior(punto.get_center(), origen), buff=0.12))
        self.play(Rotate(grupo, PI, about_point=origen), run_time=2.4)
        et_i.clear_updaters()
        et_i.next_to(punto, RIGHT, buff=0.12)
        et_menos_i = MathTex("-i", color=WHITE).scale(0.65).move_to(et_i)
        self.play(FadeOut(et_i), FadeIn(et_menos_i))
        self.remove(et_h)
        et_abajo = MathTex(r"-\mathbb{H}", color=AZUL).scale(0.8)
        et_abajo.move_to(ejes.c2p(2.05, -1.65))
        self.play(FadeIn(et_abajo))

        nota = pie("Sobre ℝ no hay raíz de −1: esta clase no entra en SL(2, ℝ).")
        nota2 = MathTex(
            r"\text{Sobre }\mathbb{C},\ i\begin{pmatrix}-1&0\\0&1\end{pmatrix}"
            r"\text{ tiene det }1:\ \mathrm{PSL}(2,\mathbb{C})=\mathrm{PGL}(2,\mathbb{C})."
        ).scale(0.58)
        nota2.next_to(nota, UP, buff=0.16)
        self.play(FadeIn(nota), FadeIn(nota2))
        self.wait(2.0)
        # unidad se usa para dejar clara la escala del dibujo
        _ = unidad


def _lado_exterior(punto, origen):
    v = punto - origen
    if abs(v[1]) >= abs(v[0]):
        return UP if v[1] >= 0 else DOWN
    return RIGHT if v[0] >= 0 else LEFT


# ---------------------------------------------------------------------------
# 0.2
# ---------------------------------------------------------------------------


class MultiplicacionCompleja(Scene):
    """Multiplicar por w ≠ 0 es rotar arg(w) y luego escalar por |w|."""

    def construct(self):
        titulo = encabezado("0.2  Multiplicar por w es rotar y homoteciar")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        rho = 1.55
        theta = 48 * DEGREES
        a = rho * np.cos(theta)
        b = rho * np.sin(theta)

        radiales = VGroup(
            *[
                Line(
                    ORIGIN,
                    2.35 * np.array([np.cos(ang), np.sin(ang), 0]),
                    stroke_width=1.4,
                    color=BLUE_D,
                )
                for ang in np.linspace(0, TAU, 12, endpoint=False)
            ]
        )
        circulos = VGroup(
            *[Circle(radius=r, color=BLUE_D, stroke_width=1.3) for r in (0.85, 1.7, 2.35)]
        )
        grilla = VGroup(radiales, circulos).set_opacity(0.55)
        grilla.shift(DOWN * 0.25)

        # El origen de la grilla deja de ser ORIGIN tras el shift: trabajamos
        # respecto de ese centro.
        centro = grilla.get_center()

        def desde(x, y):
            return centro + np.array([x, y, 0.0])

        tri = Polygon(
            desde(0.28, 0.1),
            desde(1.15, 0.18),
            desde(0.48, 0.92),
            color=AMARILLO,
            fill_color=AMARILLO,
            fill_opacity=0.92,
            stroke_width=1.2,
        )
        fantasma = tri.copy().set_fill(AMARILLO, opacity=0.18).set_stroke(AMARILLO, opacity=0.35, width=1)

        flecha = Arrow(
            centro,
            desde(1.0, 0.0),
            buff=0,
            color=WHITE,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.22,
        )
        et_uno = MathTex("1").scale(0.7).next_to(flecha.get_end(), DOWN, buff=0.12)

        destino = Arrow(
            centro,
            centro + rho * np.array([np.cos(theta), np.sin(theta), 0]),
            buff=0,
            color=ORO,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.22,
        )
        destino.set_opacity(0.35)
        et_w = MathTex("w", color=ORO).scale(0.75)
        et_w.next_to(destino.get_end(), UP, buff=0.1).set_opacity(0.7)

        formula = MathTex(
            rf"w={rho:.2f}\,e^{{i\,48^\circ}}",
            r"\quad",
            r"\begin{pmatrix}a&-b\\ b&a\end{pmatrix}"
            rf"=\begin{{pmatrix}}{a:.2f}&{-b:.2f}\\ {b:.2f}&{a:.2f}\end{{pmatrix}}",
        ).scale(0.62)
        formula.to_edge(DOWN, buff=0.32)

        self.play(FadeIn(grilla), FadeIn(fantasma), FadeIn(tri), FadeIn(flecha), FadeIn(et_uno))
        self.play(FadeIn(destino), FadeIn(et_w), FadeIn(formula))
        self.wait(0.4)

        paso = Text("1.  rotación de ángulo arg w", font_size=24, color=ORO)
        paso.next_to(titulo, DOWN, buff=0.12)
        arco = Arc(
            radius=0.62,
            start_angle=0,
            angle=theta,
            arc_center=centro,
            color=ORO,
            stroke_width=3,
        )
        self.play(FadeIn(paso), FadeIn(arco))
        self.play(
            Rotate(grilla, theta, about_point=centro),
            Rotate(tri, theta, about_point=centro),
            Rotate(flecha, theta, about_point=centro),
            run_time=2.2,
        )
        et_uno.next_to(flecha.get_end(), np.array([-np.sin(theta), np.cos(theta), 0]), buff=0.12)
        self.wait(0.3)

        paso2 = Text("2.  homotecia de razón |w|", font_size=24, color=VERDE).move_to(paso)
        self.play(FadeOut(paso), FadeIn(paso2))
        self.play(
            grilla.animate.scale(rho, about_point=centro),
            tri.animate.scale(rho, about_point=centro),
            flecha.animate.scale(rho, about_point=centro),
            arco.animate.scale(rho, about_point=centro),
            run_time=2.0,
        )
        et_w.set_opacity(1)
        self.play(et_w.animate.set_opacity(1), destino.animate.set_opacity(0.9))
        cierre = pie("Esas matrices reales son exactamente las aplicaciones lineales sobre C.")
        self.play(FadeOut(formula), FadeIn(cierre))
        self.wait(1.6)


class Conjugacion(Scene):
    """z ↦ conjugado(z) es la reflexión en el eje real, de determinante −1."""

    def construct(self):
        titulo = encabezado("0.2  La conjugación es una reflexión")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        ejes = Axes(
            x_range=[-3.4, 3.4, 1],
            y_range=[-2.4, 2.4, 1],
            x_length=8.2,
            y_length=5.6,
            tips=False,
            axis_config={"stroke_color": GRIS, "stroke_width": 2, "include_ticks": False},
        )
        ejes.shift(DOWN * 0.2 + LEFT * 0.4)
        origen = ejes.c2p(0, 0)

        tri = Polygon(
            ejes.c2p(0.35, 0.45),
            ejes.c2p(1.7, 0.55),
            ejes.c2p(0.7, 1.55),
            color=AMARILLO,
            fill_color=AMARILLO,
            fill_opacity=0.9,
            stroke_width=1.5,
        )
        # Sentido positivo, para ver que la orientación se invierte.
        centro_tri = np.mean([ejes.c2p(0.35, 0.45), ejes.c2p(1.7, 0.55), ejes.c2p(0.7, 1.55)], axis=0)
        unidad = np.linalg.norm(ejes.c2p(1, 0) - ejes.c2p(0, 0))
        sentido = Arc(
            radius=0.28 * unidad,
            start_angle=200 * DEGREES,
            angle=220 * DEGREES,
            arc_center=centro_tri,
            color="#1a1a1a",
            stroke_width=3,
        ).add_tip(tip_length=0.12)

        z = 1.15 + 1.05j
        punto = Dot(ejes.c2p(z.real, z.imag), color=WHITE, radius=0.06)
        et = MathTex("z", color=WHITE).scale(0.7).next_to(punto, UP, buff=0.1)
        guia = DashedLine(
            ejes.c2p(z.real, z.imag), ejes.c2p(z.real, -z.imag), color=GRIS, stroke_width=2
        )

        formula = MathTex(
            r"z\mapsto \overline{z}",
            r"\qquad",
            r"\begin{pmatrix}1&0\\0&-1\end{pmatrix}",
            r"\qquad \det=-1",
        ).scale(0.72)
        formula.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(ejes), FadeIn(tri), FadeIn(sentido), FadeIn(punto), FadeIn(et))
        self.play(FadeIn(formula))
        self.wait(0.3)
        grupo = VGroup(tri, sentido, punto)
        self.play(
            grupo.animate.apply_matrix([[1, 0], [0, -1]], about_point=origen),
            run_time=2.2,
        )
        self.play(FadeIn(guia))
        et_bar = MathTex(r"\overline{z}", color=WHITE).scale(0.7).next_to(punto, DOWN, buff=0.1)
        self.play(FadeOut(et), FadeIn(et_bar))
        nota = pie("Toda aplicación lineal real del plano es z ↦ αz + β z-barra.")
        self.play(FadeIn(nota.shift(UP * 0.45)))
        self.wait(1.6)


class Conformidad(Scene):
    """Df = λA con A ortogonal si y solo si se conservan los ángulos."""

    def construct(self):
        titulo = encabezado("0.2  Conforme significa que Df multiplica longitudes y gira")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        centro = DOWN * 0.35

        def marco():
            v1 = Arrow(centro, centro + RIGHT * 1.25, buff=0, color=AMARILLO, stroke_width=4)
            v2 = Arrow(centro, centro + UP * 1.25, buff=0, color=TEAL_C, stroke_width=4)
            circ = Circle(radius=1.25, color=AZUL, stroke_width=3).move_to(centro)
            sector = sector_de(centro, 0, PI / 2, 0.42, ORO)
            return VGroup(circ, v1, v2, sector)

        base = marco()
        self.play(FadeIn(base))
        et = MathTex(r"90^\circ", color=ORO).scale(0.7)
        et.next_to(centro, UR, buff=0.15)
        self.play(FadeIn(et))
        self.wait(0.3)

        # --- conforme: rotación-homotecia ---
        ang = 42 * DEGREES
        lam = 1.28
        a, b = lam * np.cos(ang), lam * np.sin(ang)
        M_conf = [[a, -b], [b, a]]
        leyenda = MathTex(
            r"Df=\lambda A,\quad A\in\mathrm{SO}(2)", color=VERDE
        ).scale(0.75)
        leyenda.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(leyenda))
        self.play(
            base.animate.apply_matrix(M_conf, about_point=centro),
            et.animate.shift(0.35 * RIGHT + 0.45 * UP),
            run_time=2.2,
        )
        sigue = pie("El círculo sigue siendo círculo y el ángulo sigue siendo 90°.")
        sigue.next_to(leyenda, UP, buff=0.12)
        self.play(FadeIn(sigue))
        self.wait(1.3)

        # --- no conforme ---
        self.play(FadeOut(base), FadeOut(et), FadeOut(sigue), FadeOut(leyenda))
        base = marco()
        et = MathTex(r"90^\circ", color=ORO).scale(0.7).next_to(centro, UR, buff=0.15)
        self.play(FadeIn(base), FadeIn(et))
        M_mal = [[1.65, 0.55], [0.2, 0.72]]
        v1 = np.array([1.25, 0.0])
        v2 = np.array([0.0, 1.25])
        w1 = np.array(M_mal) @ v1
        w2 = np.array(M_mal) @ v2
        nuevo = angulo_entre(w1, w2)
        leyenda = MathTex(
            rf"L=\begin{{pmatrix}}1.65&0.55\\0.20&0.72\end{{pmatrix}}"
            rf"\quad {nuevo:.0f}^\circ",
            color=ROJO,
        ).scale(0.7)
        leyenda.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(leyenda))
        self.play(base.animate.apply_matrix(M_mal, about_point=centro), FadeOut(et), run_time=2.2)
        # Ángulo verdadero entre las imágenes, medido en el dibujo.
        # Las flechas son las submobjects 1 y 2.
        flecha1, flecha2 = base[1], base[2]
        angulo = Angle(flecha1, flecha2, radius=0.48, color=ROJO, stroke_width=3)
        et2 = MathTex(rf"{nuevo:.0f}^\circ", color=ROJO).scale(0.7)
        et2.next_to(angulo, UR, buff=0.08)
        self.play(FadeIn(angulo), FadeIn(et2))
        sigue = pie("Hay orientación (det > 0) pero el círculo se volvió elipse: no es conforme.")
        sigue.next_to(leyenda, UP, buff=0.12)
        self.play(FadeIn(sigue))
        self.wait(1.5)

        # --- anticonforme ---
        self.play(FadeOut(base), FadeOut(angulo), FadeOut(et2), FadeOut(sigue), FadeOut(leyenda))
        base = marco()
        self.play(FadeIn(base))
        leyenda = MathTex(
            r"\overline{z}:\quad A\in\mathrm{O}(2),\ \det A=-1", color=TEAL_C
        ).scale(0.75)
        leyenda.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(leyenda))
        self.play(
            base.animate.apply_matrix([[1, 0], [0, -1]], about_point=centro),
            run_time=1.8,
        )
        sigue = pie("El ángulo no orientado se conserva y la orientación se invierte: anticonforme.")
        sigue.next_to(leyenda, UP, buff=0.12)
        self.play(FadeIn(sigue))
        self.wait(1.8)


def sector_de(centro, a0, a1, radio, color):
    arc = Arc(radius=radio, start_angle=a0, angle=a1 - a0, arc_center=centro, color=color, stroke_width=3)
    return arc


class HolomorfaYAngulo(Scene):
    """En dimensión 2, conforme y que preserva orientación equivale a holomorfa con f' ≠ 0."""

    def construct(self):
        titulo = encabezado("0.2  En el plano, holomorfa con f' distinta de 0 es conforme")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        z0 = 0.55 + 0.8j
        d1 = np.array([1.0, 0.0])
        d2 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])

        dom = _ejes_iguales([-0.45, 1.85], [-0.3, 1.85], 4.5).shift(LEFT * 3.35 + DOWN * 0.35)
        cod = _ejes_iguales([-1.35, 1.7], [-0.45, 2.05], 4.7).shift(RIGHT * 3.15 + DOWN * 0.35)

        et_d = Text("dominio", font_size=22, color=GRIS).next_to(dom, UP, buff=0.12)
        et_c = Text("imagen", font_size=22, color=GRIS).next_to(cod, UP, buff=0.12)

        self.play(FadeIn(dom), FadeIn(cod), FadeIn(et_d), FadeIn(et_c))

        colores = [AMARILLO, TEAL_C]
        muestras = [
            [z0 + (t * 0.5) * complex(float(d[0]), float(d[1])) for t in np.linspace(0, 1, 18)]
            for d in (d1, d2)
        ]
        dom_mob = _haz(dom, z0, muestras, colores)
        arco = _arco_coord(dom, z0, 0.0, np.pi / 3, 0.32, ORO)
        et_ang = MathTex(r"60^\circ", color=ORO).scale(0.6)
        et_ang.next_to(dom.c2p(z0.real, z0.imag), UR, buff=0.05)
        f_tex = MathTex(r"f(z)=z^2", color=VERDE).scale(0.75)
        f_tex.move_to(UP * 2.35)
        self.play(FadeIn(dom_mob), FadeIn(arco), FadeIn(et_ang), FadeIn(f_tex))

        # Imagen por z^2 de las curvas, y tangentes rotadas por arg f'(z0).
        fp = 2 * z0  # f'(z0)
        arg = np.angle(fp)
        z_img = z0 * z0
        cod_mob = _haz(cod, z_img, [[z * z for z in rama] for rama in muestras], colores)
        arco2 = _arco_coord(cod, z_img, arg, arg + np.pi / 3, 0.34, ORO)
        et2 = MathTex(r"60^\circ", color=ORO).scale(0.6)
        et2.next_to(cod.c2p(z_img.real, z_img.imag), UL, buff=0.05)
        escala = MathTex(rf"|f'(z_0)|={abs(fp):.2f}", color=GRIS).scale(0.6)
        escala.next_to(f_tex, DOWN, buff=0.1)
        self.play(FadeIn(cod_mob), FadeIn(arco2), FadeIn(et2), FadeIn(escala), run_time=1.6)
        nota = pie("Las dos tangentes giran el ángulo de f'(z0). El de 60 grados se conserva.")
        self.play(FadeIn(nota))
        self.wait(1.5)

        # Mapa real diferenciable que no es holomorfo.
        self.play(FadeOut(cod_mob), FadeOut(arco2), FadeOut(et2), FadeOut(escala), FadeOut(nota), FadeOut(f_tex))
        g_tex = MathTex(r"g(x,y)=(1.7x,\ 0.65y)", color=ROJO).scale(0.72).move_to(UP * 2.35)
        self.play(FadeIn(g_tex))

        def g(z):
            return complex(1.7 * z.real, 0.65 * z.imag)

        cod_mob = _haz(cod, g(z0), [[g(z) for z in rama] for rama in muestras], colores)
        # Direcciones imagen de la diferencial.
        J1 = np.array([1.7, 0.0])
        J2 = np.array([1.7 * d2[0], 0.65 * d2[1]])
        ang_g = angulo_entre(J1, J2)
        a1 = np.arctan2(J1[1], J1[0])
        a2 = np.arctan2(J2[1], J2[0])
        # Ordenar para que el arco vaya por el sector menor positivo.
        if a2 < a1:
            a1, a2 = a2, a1
        z_g = g(z0)
        arco3 = _arco_coord(cod, z_g, a1, a2, 0.34, ROJO)
        et3 = MathTex(rf"{ang_g:.0f}^\circ", color=ROJO).scale(0.62)
        et3.next_to(cod.c2p(z_g.real, z_g.imag), UR, buff=0.06)
        self.play(FadeIn(cod_mob), FadeIn(arco3), FadeIn(et3))
        nota = pie("Dg no es multiplicación por un complejo: el ángulo de 60 grados no se conserva.")
        self.play(FadeIn(nota))
        self.wait(1.8)


def _ejes_iguales(x_range, y_range, x_length):
    span_x = x_range[1] - x_range[0]
    span_y = y_range[1] - y_range[0]
    return Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=x_length,
        y_length=x_length * span_y / span_x,
        tips=False,
        axis_config={"stroke_color": GRIS, "stroke_width": 1.6, "include_ticks": False},
    )


def _haz(axes, z0, ramas, colores):
    """Curvas desde z0 y flechas tangentes, a partir de muestras complejas."""
    p = axes.c2p(z0.real, z0.imag)
    unidad = np.linalg.norm(axes.c2p(1, 0) - axes.c2p(0, 0))
    curvas = VGroup()
    flechas = VGroup()
    for rama, col in zip(ramas, colores):
        pts = [axes.c2p(w.real, w.imag) for w in rama]
        mob = VMobject(color=col, stroke_width=4)
        mob.set_points_smoothly(pts)
        curvas.add(mob)
        direccion = pts[1] - pts[0]
        direccion = direccion / np.linalg.norm(direccion)
        flechas.add(
            Arrow(
                p,
                p + 0.42 * unidad * direccion,
                buff=0,
                color=col,
                stroke_width=3.5,
                max_tip_length_to_length_ratio=0.25,
            )
        )
    return VGroup(curvas, flechas, Dot(p, color=WHITE, radius=0.055))


def _arco_coord(axes, z, a0, a1, radio_coord, color):
    unidad = np.linalg.norm(axes.c2p(1, 0) - axes.c2p(0, 0))
    return Arc(
        radius=radio_coord * unidad,
        start_angle=a0,
        angle=a1 - a0,
        arc_center=axes.c2p(z.real, z.imag),
        color=color,
        stroke_width=3,
    )


# ---------------------------------------------------------------------------
# 0.3
# ---------------------------------------------------------------------------


class CorteMeridiano(Scene):
    """Figura 1 de las notas: corte de la proyección estereográfica, y N ↔ ∞."""

    def construct(self):
        titulo = encabezado("0.3  Proyección estereográfica, corte meridiano")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        R = 1.85
        centro = LEFT * 2.55 + DOWN * 0.2
        alpha = ValueTracker(155 * DEGREES)

        def local(a):
            return np.array([np.sin(a), np.cos(a), 0.0])

        def punto_esfera():
            return centro + R * local(alpha.get_value())

        def y_plano():
            a = alpha.get_value()
            return float(np.cos(a / 2) / np.sin(a / 2))

        def punto_plano():
            return centro + np.array([R * y_plano(), 0.0, 0.0])

        norte = centro + R * UP
        circulo = Circle(radius=R, color=AZUL, stroke_width=3.5).move_to(centro)
        circulo.set_fill(AZUL, opacity=0.06)
        ecuador = Line(centro + LEFT * 1.15, centro + RIGHT * 8.3, color=TEAL_C, stroke_width=2.5)
        n_dot = Dot(norte, color=ROJO, radius=0.07)
        et_n = MathTex("N", color=ROJO).scale(0.7).next_to(n_dot, UP, buff=0.1)
        et_inf = MathTex(r"\infty", color=ROJO).scale(0.6).next_to(et_n, RIGHT, buff=0.08)

        rayo = always_redraw(
            lambda: Line(norte, punto_plano(), color=ORO, stroke_width=2.6)
        )
        p_dot = always_redraw(lambda: Dot(punto_esfera(), color=VERDE, radius=0.065))
        pi_dot = always_redraw(lambda: Dot(punto_plano(), color=AMARILLO, radius=0.065))
        et_p = Text("P", font_size=22, color=VERDE).next_to(punto_esfera(), LEFT, buff=0.1)
        et_pi = MathTex(r"\pi(P)", color=AMARILLO).scale(0.55).next_to(punto_plano(), DOWN, buff=0.12)
        et_p.add_updater(lambda m: m.next_to(punto_esfera(), LEFT, buff=0.1))
        et_pi.add_updater(lambda m: m.next_to(punto_plano(), DOWN, buff=0.12))

        y_num = DecimalNumber(y_plano(), num_decimal_places=2, include_sign=True, font_size=30, color=AMARILLO)
        y_num.add_updater(lambda m: m.set_value(y_plano()))
        norma = DecimalNumber(1.0, num_decimal_places=3, font_size=30, color=VERDE)
        norma.add_updater(lambda m: m.set_value(_norma_inversa(y_plano())))

        formula = MathTex(r"\pi(x)=\frac{x_1}{1-x_{n+1}}").scale(0.68)
        y_label = MathTex(r"y=", color=AMARILLO).scale(0.7)
        norma_label = MathTex(r"|\pi^{-1}(y)|=", color=VERDE).scale(0.7)
        fila_y = VGroup(y_label, y_num).arrange(RIGHT, buff=0.12)
        fila_n = VGroup(norma_label, norma).arrange(RIGHT, buff=0.12)
        panel = VGroup(formula, fila_y, fila_n).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        panel.to_corner(UR, buff=0.4).shift(DOWN * 0.55 + LEFT * 0.1)
        # Los DecimalNumber se mueven al actualizar; los fijamos al sitio del panel.
        y_ancla = y_num.get_center()
        n_ancla = norma.get_center()
        y_num.add_updater(lambda m, p=y_ancla: m.move_to(p))
        norma.add_updater(lambda m, p=n_ancla: m.move_to(p))

        self.play(
            FadeIn(circulo),
            FadeIn(ecuador),
            FadeIn(n_dot),
            FadeIn(et_n),
            FadeIn(et_inf),
            FadeIn(panel),
        )
        self.add(rayo, p_dot, pi_dot, et_p, et_pi)
        self.wait(0.4)
        self.play(alpha.animate.set_value(28 * DEGREES), run_time=6.5, rate_func=lambda t: t)
        nota = pie("Cerca de N la proyección se va a infinito. N es el punto del infinito.")
        self.play(FadeIn(nota))
        self.wait(0.6)

        # Coda: coordenadas homogéneas, el puente con el capítulo 1.
        et_p.clear_updaters()
        et_pi.clear_updaters()
        self.play(FadeOut(nota), FadeOut(et_p), FadeOut(et_pi))
        coda = MathTex(
            r"\widehat{\mathbb{C}}\cong\mathbb{CP}^1",
            r"\quad",
            r"z=[z:1],\quad \infty=[1:0]",
        ).scale(0.64)
        coda.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(coda))
        self.wait(2.0)


def _norma_inversa(y: float) -> float:
    den = y * y + 1.0
    x1 = 2 * y / den
    x2 = (y * y - 1.0) / den
    return float(np.sqrt(x1 * x1 + x2 * x2))


class FactorConforme(Scene):
    """El factor 2/(1+|y|^2) depende del punto: hay ángulos, no hay isometría."""

    def construct(self):
        titulo = encabezado("0.3  Conforme, pero no isometría")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        R = 2.05
        centro = LEFT * 3.15 + DOWN * 0.15
        circulo = Circle(radius=R, color=AZUL, stroke_width=3).move_to(centro)
        circulo.set_fill(AZUL, opacity=0.05)
        norte = centro + R * UP
        ecuador = Line(centro + LEFT * 0.9, centro + RIGHT * 9.2, color=TEAL_C, stroke_width=2.4)
        n_dot = Dot(norte, color=ROJO, radius=0.06)
        et_n = MathTex("N", color=ROJO).scale(0.6).next_to(n_dot, UP, buff=0.08)

        def arco(a0, a1, color):
            # α desde el norte; el ángulo estándar es π/2 − α.
            phi0 = PI / 2 - a0
            phi1 = PI / 2 - a1
            return Arc(
                radius=R,
                start_angle=phi0,
                angle=phi1 - phi0,
                arc_center=centro,
                color=color,
                stroke_width=7,
            )

        def y_de(a):
            return float(np.cos(a / 2) / np.sin(a / 2))

        def punto(a):
            return centro + R * np.array([np.sin(a), np.cos(a), 0.0])

        def en_eje(y):
            return centro + np.array([R * y, 0.0, 0.0])

        # Dos arcos de la misma longitud angular.
        delta = 16 * DEGREES
        sur0, sur1 = 172 * DEGREES, 172 * DEGREES - delta
        nor0, nor1 = 58 * DEGREES, 58 * DEGREES - delta
        arco_sur = arco(sur0, sur1, AMARILLO)
        arco_nor = arco(nor0, nor1, AMARILLO)

        def rayos(a0, a1):
            return VGroup(
                DashedLine(norte, en_eje(y_de(a0)), color=GRIS, stroke_width=1.6),
                DashedLine(norte, en_eje(y_de(a1)), color=GRIS, stroke_width=1.6),
            )

        seg_sur = Line(en_eje(y_de(sur1)), en_eje(y_de(sur0)), color=AMARILLO, stroke_width=6)
        seg_nor = Line(en_eje(y_de(nor1)), en_eje(y_de(nor0)), color=AMARILLO, stroke_width=6)

        long_esf = delta  # radio 1
        long_sur = abs(y_de(sur0) - y_de(sur1))
        long_nor = abs(y_de(nor0) - y_de(nor1))

        self.play(FadeIn(circulo), FadeIn(ecuador), FadeIn(n_dot), FadeIn(et_n))
        self.play(FadeIn(arco_sur), FadeIn(arco_nor))
        self.play(FadeIn(rayos(sur0, sur1)), FadeIn(rayos(nor0, nor1)))
        self.play(FadeIn(seg_sur), FadeIn(seg_nor))

        lab_sur = MathTex(rf"{long_sur:.2f}", color=AMARILLO).scale(0.55)
        lab_sur.next_to(seg_sur, DOWN, buff=0.12)
        lab_nor = MathTex(rf"{long_nor:.2f}", color=AMARILLO).scale(0.55)
        lab_nor.next_to(seg_nor, DOWN, buff=0.12)
        lab_esf = MathTex(rf"\text{{arco }}{long_esf:.2f}", color=AMARILLO).scale(0.55)
        lab_esf.next_to(arco_sur, LEFT, buff=0.12)
        self.play(FadeIn(lab_sur), FadeIn(lab_nor), FadeIn(lab_esf))

        formula = MathTex(
            r"ds^{2}=\frac{4\,|dy|^{2}}{(1+|y|^{2})^{2}}"
        ).scale(0.7)
        formula.to_corner(UR, buff=0.4).shift(DOWN * 0.5)
        nota = pie("Misma longitud en la esfera, longitudes distintas en el plano.")
        self.play(FadeIn(formula), FadeIn(nota))
        self.wait(2.2)
        _ = punto


class MetricaCordal(Scene):
    """χ(z, w) es la cuerda euclídea en R^3 entre las preimágenes."""

    def construct(self):
        titulo = encabezado("0.3  La métrica cordal deja el infinito a distancia finita")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        R = 1.9
        centro = LEFT * 2.7 + DOWN * 0.15
        circulo = Circle(radius=R, color=AZUL, stroke_width=3.2).move_to(centro)
        circulo.set_fill(AZUL, opacity=0.05)
        ecuador = Line(centro + LEFT * 1.0, centro + RIGHT * 8.6, color=TEAL_C, stroke_width=2.4)
        norte = centro + R * UP
        sur = centro + R * DOWN
        n_dot = Dot(norte, color=ROJO, radius=0.06)
        s_dot = Dot(sur, color=VERDE, radius=0.065)
        o_dot = Dot(centro, color=WHITE, radius=0.05)
        et_n = MathTex(r"N=\infty", color=ROJO).scale(0.55).next_to(n_dot, UP, buff=0.08)
        et_0 = MathTex("0", color=VERDE).scale(0.55).next_to(s_dot, DOWN, buff=0.08)
        et_o = MathTex("0", color=WHITE).scale(0.5).next_to(o_dot, DOWN, buff=0.1)

        y = ValueTracker(0.7)

        def lift_local(val):
            den = val * val + 1.0
            return np.array([2 * val / den, (val * val - 1.0) / den, 0.0])

        def lift():
            return centro + R * lift_local(y.get_value())

        def plano():
            return centro + np.array([R * y.get_value(), 0.0, 0.0])

        def chi():
            val = y.get_value()
            return 2 * abs(val) / np.sqrt(1.0 + val * val)

        cuerda = always_redraw(lambda: Line(sur, lift(), color=ROJO, stroke_width=3.5))
        rayo = always_redraw(lambda: Line(norte, plano(), color=ORO, stroke_width=1.8))
        movil = always_redraw(lambda: Dot(lift(), color=AMARILLO, radius=0.06))
        y_dot = always_redraw(lambda: Dot(plano(), color=AMARILLO, radius=0.055))

        chi_num = DecimalNumber(chi(), num_decimal_places=2, font_size=30, color=ROJO)
        y_num = DecimalNumber(y.get_value(), num_decimal_places=2, font_size=30, color=AMARILLO)
        chi_num.add_updater(lambda m: m.set_value(chi()).move_to(chi_ancla))
        y_num.add_updater(lambda m: m.set_value(y.get_value()).move_to(y_ancla))

        chi_lab = MathTex(r"\chi(0,y)=", color=ROJO).scale(0.7)
        y_lab = MathTex(r"|y-0|=", color=AMARILLO).scale(0.7)
        fila_c = VGroup(chi_lab, chi_num).arrange(RIGHT, buff=0.12)
        fila_y = VGroup(y_lab, y_num).arrange(RIGHT, buff=0.12)
        formula = MathTex(
            r"\chi(z,w)=\frac{2|z-w|}{\sqrt{(1+|z|^2)(1+|w|^2)}}"
        ).scale(0.58)
        panel = VGroup(formula, fila_c, fila_y).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        panel.to_corner(UR, buff=0.38).shift(DOWN * 0.55)
        chi_ancla = chi_num.get_center()
        y_ancla = y_num.get_center()

        self.play(
            FadeIn(circulo), FadeIn(ecuador), FadeIn(n_dot), FadeIn(s_dot),
            FadeIn(o_dot), FadeIn(et_n), FadeIn(et_0), FadeIn(et_o), FadeIn(panel),
        )
        self.add(rayo, cuerda, movil, y_dot)
        self.wait(0.5)
        self.play(y.animate.set_value(5.2), run_time=6, rate_func=lambda t: t)
        nota = pie("La distancia euclídea se va a infinito; la cuerda en la esfera se queda ≤ 2.")
        self.play(FadeIn(nota))
        self.wait(1.8)


class Estereografica3D(ThreeDScene):
    """La proyección en la esfera, y un ángulo que sobrevive en el plano."""

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-42 * DEGREES, zoom=0.92)

        titulo = Text("0.3  De la esfera al plano", font_size=32, color=WHITE)
        titulo.to_edge(UP, buff=0.25)
        self.add_fixed_in_frame_mobjects(titulo)

        esfera = Sphere(radius=1.0, resolution=(28, 28))
        esfera.set_color(BLUE_D)
        esfera.set_opacity(0.28)

        plano = NumberPlane(
            x_range=[-3.2, 3.2, 1],
            y_range=[-3.2, 3.2, 1],
            background_line_style={"stroke_color": TEAL, "stroke_width": 1, "stroke_opacity": 0.35},
            axis_config={"stroke_color": TEAL, "stroke_width": 1.5, "stroke_opacity": 0.5},
            faded_line_ratio=0,
        )
        ecuador = Circle(radius=1, color=TEAL_C, stroke_width=2.5)

        norte = np.array([0.0, 0.0, 1.0])
        n_dot = Dot3D(point=norte, radius=0.045, color=RED)
        etiqueta_n = Text("N", font_size=22, color=RED)
        self.add_fixed_in_frame_mobjects(etiqueta_n)
        # La etiqueta fija se coloca una vez, arriba a la izquierda del título no:
        # mejor un rótulo de leyenda, no pegado al polo (la cámara se mueve).
        etiqueta_n.next_to(titulo, DOWN, buff=0.05).align_to(titulo, LEFT)
        etiqueta_n.become(Text("polo norte = infinito", font_size=22, color=RED).next_to(titulo, DOWN, buff=0.08).align_to(titulo, LEFT))

        theta = ValueTracker(2.15)
        phi = 0.55

        def punto_en_esfera():
            th = theta.get_value()
            return np.array(
                [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)]
            )

        def proyeccion():
            p = punto_en_esfera()
            return np.array([p[0] / (1 - p[2]), p[1] / (1 - p[2]), 0.0])

        p_dot = Dot3D(point=punto_en_esfera(), radius=0.04, color=GREEN)
        q_dot = Dot3D(point=proyeccion(), radius=0.04, color=YELLOW)
        p_dot.add_updater(lambda m: m.move_to(punto_en_esfera()))
        q_dot.add_updater(lambda m: m.move_to(proyeccion()))
        rayo = always_redraw(lambda: Line3D(norte, proyeccion(), color=YELLOW, thickness=0.012))

        self.play(FadeIn(plano), FadeIn(esfera), FadeIn(ecuador), FadeIn(n_dot), run_time=1.2)
        self.add(rayo, p_dot, q_dot)
        self.wait(0.4)
        self.play(theta.animate.set_value(0.72), run_time=4.5, rate_func=lambda t: t)
        self.wait(0.4)

        # Paralelo y meridiano que se cortan en ángulo recto, y sus proyecciones.
        z0 = 0.35
        ph = 32 * DEGREES
        r_par = float(np.sqrt(1 - z0 * z0))
        paralelo = ParametricFunction(
            lambda t: np.array([r_par * np.cos(t), r_par * np.sin(t), z0]),
            t_range=[0, TAU],
            color=TEAL_C,
            stroke_width=3.5,
        )
        meridiano = ParametricFunction(
            lambda t: np.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)]),
            t_range=[0.18, PI - 0.02],
            color=AMARILLO,
            stroke_width=3.5,
        )
        radio_proy = r_par / (1 - z0)
        circulo_proy = Circle(radius=radio_proy, color=TEAL_C, stroke_width=3).shift(OUT * 0)
        # Circle ya vive en z = 0.
        direccion = np.array([np.cos(ph), np.sin(ph), 0.0])
        recta_proy = Line3D(-2.5 * direccion, 2.5 * direccion, color=AMARILLO, thickness=0.01)

        cruce = np.array([r_par * np.cos(ph), r_par * np.sin(ph), z0])
        cruce_plano = np.array(
            [cruce[0] / (1 - z0), cruce[1] / (1 - z0), 0.0]
        )
        # Tangentes en la esfera.
        d_mer = np.array([z0 * np.cos(ph), z0 * np.sin(ph), -r_par])
        d_par = np.array([-np.sin(ph), np.cos(ph), 0.0])
        marca_esf = _marca3d(cruce, d_mer, d_par, 0.16, WHITE)
        d_rayo = direccion
        d_tang = np.array([-np.sin(ph), np.cos(ph), 0.0])
        marca_plano = _marca3d(cruce_plano, d_rayo, d_tang, 0.22, WHITE)

        self.play(FadeOut(rayo), FadeOut(p_dot), FadeOut(q_dot))
        self.play(FadeIn(paralelo), FadeIn(meridiano), FadeIn(circulo_proy), FadeIn(recta_proy))
        self.play(FadeIn(marca_esf), FadeIn(marca_plano))
        leyenda = Text(
            "Meridiano y paralelo se cortan a 90°. También sus proyecciones.",
            font_size=22,
            color=GRIS,
        )
        leyenda.to_edge(DOWN, buff=0.25)
        self.add_fixed_in_frame_mobjects(leyenda)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(5.5)
        self.stop_ambient_camera_rotation()
        self.wait(0.6)


def _marca3d(origen, d1, d2, size, color):
    a = np.array(d1, dtype=float)
    b = np.array(d2, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    p1 = origen + size * a
    p2 = origen + size * a + size * b
    p3 = origen + size * b
    return VGroup(
        Line3D(p1, p2, color=color, thickness=0.008),
        Line3D(p2, p3, color=color, thickness=0.008),
    )


# ---------------------------------------------------------------------------
# 0.4 y 0.5
# ---------------------------------------------------------------------------


class OrbitaEstabilizador(Scene):
    """Órbita, estabilizador, acción libre y el cociente G/H."""

    def construct(self):
        titulo = encabezado("0.4  Órbita y estabilizador")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        hexa = RegularPolygon(n=6, radius=2.15, color=AZUL, stroke_width=3.5)
        hexa.shift(DOWN * 0.25)
        centro = hexa.get_center()
        vertices = hexa.get_vertices()
        fantasmas = VGroup(
            *[Dot(v, radius=0.07, color=AMARILLO).set_opacity(0.2) for v in vertices]
        )
        viajero = Dot(vertices[0], radius=0.085, color=AMARILLO)
        et_x = MathTex("x", color=AMARILLO).scale(0.65).next_to(viajero, RIGHT, buff=0.1)
        et_x.add_updater(lambda m: m.next_to(viajero, RIGHT, buff=0.1))

        self.play(FadeIn(hexa), FadeIn(fantasmas), FadeIn(viajero), FadeIn(et_x))
        paso = Text("G = rotaciones de 60°", font_size=24, color=GRIS)
        paso.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(paso))
        for k in range(1, 6):
            self.play(Rotate(viajero, 60 * DEGREES, about_point=centro), run_time=0.42)
            self.play(fantasmas[k].animate.set_opacity(1), run_time=0.12)
        self.play(fantasmas[0].animate.set_opacity(1), run_time=0.2)

        texto = Text(
            "Una sola órbita sobre los vértices, y solo la identidad fija x.",
            font_size=24,
            color=GRIS,
        )
        texto.to_edge(DOWN, buff=0.38)
        self.play(FadeOut(paso), FadeIn(texto))
        self.wait(1.2)

        centro_dot = Dot(centro, color=ROJO, radius=0.08)
        et_o = MathTex("O", color=ROJO).scale(0.6).next_to(centro_dot, UP, buff=0.12)
        self.play(FadeIn(centro_dot), FadeIn(et_o))
        self.play(Rotate(viajero, 120 * DEGREES, about_point=centro), run_time=1.1)
        texto2 = Text(
            "El centro no se mueve: su órbita es un punto y su estabilizador es G.",
            font_size=24,
            color=GRIS,
        )
        texto2.to_edge(DOWN, buff=0.38)
        self.play(FadeOut(texto), FadeIn(texto2))
        self.wait(0.6)
        teo = MathTex(
            r"G/\mathrm{Stab}(x)\;\longleftrightarrow\;\text{órbita de }x"
        ).scale(0.7)
        teo.next_to(texto2, UP, buff=0.16)
        self.play(FadeIn(teo))
        self.wait(1.8)


class EstabilizadorYDimension(Scene):
    """El estabilizador de ∞ es el grupo de semejanzas, y de ahí sale la dimensión 6."""

    def construct(self):
        titulo = encabezado("0.4–0.5  Quién fija el infinito, y por qué hay dimensión 6")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        ejes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.2, 2.2, 1],
            x_length=6.4,
            y_length=4.6,
            tips=False,
            axis_config={"stroke_color": GRIS, "stroke_width": 1.8, "include_ticks": False},
        )
        ejes.shift(LEFT * 2.15 + DOWN * 0.35)
        origen = ejes.c2p(0, 0)
        tri = Polygon(
            ejes.c2p(0.3, 0.25),
            ejes.c2p(1.25, 0.3),
            ejes.c2p(0.55, 1.15),
            color=AMARILLO,
            fill_color=AMARILLO,
            fill_opacity=0.9,
            stroke_width=1.2,
        )
        infinito = VGroup(
            Circle(radius=0.32, color=ROJO, stroke_width=2.5),
            MathTex(r"\infty", color=ROJO).scale(0.7),
        )
        infinito.move_to(ejes.c2p(0, 2.05) + UP * 0.15)
        et_fijo = Text("fijo", font_size=20, color=ROJO).next_to(infinito, RIGHT, buff=0.12)

        contador = Integer_panel()
        contador.to_edge(RIGHT, buff=0.55).shift(UP * 0.2)

        self.play(FadeIn(ejes), FadeIn(tri), FadeIn(infinito), FadeIn(et_fijo), FadeIn(contador))
        self.wait(0.3)

        # Traslación: 2 parámetros.
        original = tri.copy()
        self.add(original.set_opacity(0.25))
        paso = Text("traslaciones  +2", font_size=24, color=VERDE)
        paso.next_to(titulo, DOWN, buff=0.1)
        self.play(FadeIn(paso))
        self.play(tri.animate.shift(RIGHT * 0.7 + UP * 0.35), run_time=1.3)
        contador._poner(2)
        self.wait(0.35)
        self.wait(0.3)

        self.play(tri.animate.move_to(original.get_center()), FadeOut(original), run_time=0.6)
        paso2 = Text("rotación  +1", font_size=24, color=VERDE).move_to(paso)
        self.play(FadeOut(paso), FadeIn(paso2))
        self.play(Rotate(tri, 35 * DEGREES, about_point=origen), run_time=1.2)
        contador._poner(3)
        self.wait(0.35)

        paso3 = Text("homotecia  +1", font_size=24, color=VERDE).move_to(paso)
        self.play(FadeOut(paso2), FadeIn(paso3))
        self.play(tri.animate.scale(1.35, about_point=origen), run_time=1.2)
        contador._poner(4)
        self.wait(0.35)
        stab = MathTex(r"\mathrm{Stab}(\infty)=\{z\mapsto az+b\}", color=GRIS).scale(0.62)
        stab.next_to(contador, DOWN, buff=0.35)
        self.play(FadeIn(stab))
        self.wait(0.5)

        paso4 = Text("la órbita, el propio plano  +2", font_size=24, color=ORO).move_to(paso)
        self.play(FadeOut(paso3), FadeIn(paso4))
        punto = Dot(origen, color=ORO, radius=0.07)
        self.play(FadeIn(punto))
        self.play(punto.animate.move_to(ejes.c2p(1.4, -0.7)), run_time=0.7)
        self.play(punto.animate.move_to(ejes.c2p(-1.1, 0.5)), run_time=0.7)
        contador._poner(6)
        self.wait(0.4)
        cierre = MathTex(
            r"\dim\mathrm{PSL}(2,\mathbb{C})=6=\dim\mathrm{O}(3,1)"
        ).scale(0.68)
        cierre.to_edge(DOWN, buff=0.32)
        self.play(FadeIn(cierre))
        self.wait(2.0)


class _Contador(VGroup):
    def __init__(self):
        super().__init__()
        self.valor = 0
        self.etiqueta = Text("dim", font_size=24, color=GRIS)
        self.numero = Text("0", font_size=54, color=WHITE)
        self.numero.next_to(self.etiqueta, DOWN, buff=0.12)
        self.add(self.etiqueta, self.numero)

    def _poner(self, n):
        nuevo = Text(str(n), font_size=54, color=AMARILLO if n == 6 else WHITE)
        nuevo.move_to(self.numero)
        self.numero.become(nuevo)
        self.valor = n


def Integer_panel():
    return _Contador()


# Manim pide que animate_set exista como método que devuelva una animación.
# Lo resolvemos con un método de escena que usa UpdateFromAlphaFunc, definido aquí
# como animación explícita en la escena de arriba mediante .animate no estándar.
# Reescribimos el llamado: ver parche abajo en la propia escena si falla.
# Para que tri.animate y contador funcionen, implementamos el cambio del contador
# con una animación propia.


def _animar_contador(contador, destino, **kwargs):
    inicio = contador.valor

    def upd(m, t):
        if t >= 1:
            m._poner(destino)
        else:
            m._poner(inicio)

    return UpdateFromAlphaFunc(contador, upd, **kwargs)


# Parche: el método usado en la escena no existe. Lo añadimos a la clase.
def _animate_set(self, n):
    return _animar_contador(self, n)


_Contador.animate_set = _animate_set


class ProgramaErlangen(Scene):
    """Una geometría es un par (G, X). Möbius conserva el ángulo, no la longitud."""

    def construct(self):
        titulo = encabezado("0.6  El programa de Erlangen")
        self.play(FadeIn(titulo, shift=DOWN * 0.1))

        par = MathTex(r"(G,X)", color=WHITE).scale(1.4)
        frase = Text("Una geometría es lo que un grupo conserva al actuar.", font_size=24, color=GRIS)
        frase.next_to(par, DOWN, buff=0.3)
        self.play(FadeIn(par), FadeIn(frase))
        self.wait(0.8)
        self.play(par.animate.scale(0.55).to_edge(UP, buff=1.05).shift(LEFT * 4.6), FadeOut(frase))

        mob = MathTex(
            r"(\mathrm{M\ddot{o}b}(n),\; S^{n}\cong\widehat{\mathbb{R}}^{n})",
            color=AMARILLO,
        ).scale(0.78)
        mob.next_to(titulo, DOWN, buff=0.18)
        self.play(FadeIn(mob))

        # Icono: círculo y recta, el mismo tipo de objeto, con un ángulo.
        icono_centro = DOWN * 0.85 + LEFT * 3.2
        circ = Circle(radius=1.15, color=AZUL, stroke_width=3.5).move_to(icono_centro)
        recta = Line(icono_centro + LEFT * 2.3, icono_centro + RIGHT * 2.3 + UP * 0.15, color=TEAL_C, stroke_width=3.5)
        self.play(FadeIn(circ), FadeIn(recta))
        et_obj = Text("círculo o recta:\nla misma familia", font_size=22, color=GRIS)
        et_obj.next_to(circ, DOWN, buff=0.35)
        self.play(FadeIn(et_obj))

        columnas = VGroup(
            _tarjeta("Analítica", "aplicaciones\nconformes", "capítulo 3"),
            _tarjeta("Algebraica", "matrices sobre\nC o Clifford", "capítulos 1 y 4"),
            _tarjeta("Hiperbólica", "isometrías\ndel espacio hiperbólico", "capítulo 5"),
        ).arrange(RIGHT, buff=0.28)
        columnas.scale(0.92).to_edge(RIGHT, buff=0.4).shift(DOWN * 0.35)
        self.play(FadeIn(columnas, lag_ratio=0.2))

        conserva = Text("se conserva el ángulo", font_size=24, color=VERDE)
        no_cons = Text("no la longitud, ni el área, ni la rectitud", font_size=24, color=ROJO)
        ley = VGroup(conserva, no_cons).arrange(DOWN, buff=0.12)
        ley.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ley))
        self.wait(2.2)


def _tarjeta(titulo, cuerpo, pie_txt):
    t = Text(titulo, font_size=24, color=AMARILLO)
    c = Text(cuerpo, font_size=20, color=WHITE, line_spacing=0.9)
    p = Text(pie_txt, font_size=18, color=GRIS)
    grupo = VGroup(t, c, p).arrange(DOWN, buff=0.14)
    marco = RoundedRectangle(
        width=grupo.width + 0.45,
        height=grupo.height + 0.4,
        corner_radius=0.12,
        color=AZUL,
        stroke_width=1.6,
    )
    grupo.move_to(marco)
    return VGroup(marco, grupo)
