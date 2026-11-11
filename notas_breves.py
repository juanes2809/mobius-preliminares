"""Animaciones de las notas breves: el grupo, sus círculos y su invariante."""

from __future__ import annotations

import numpy as np
from manim import (
    TAU,
    UL,
    Arrow,
    Circle,
    Create,
    DOWN,
    DashedLine,
    DEGREES,
    DecimalNumber,
    Dot,
    Dot3D,
    LEFT,
    Line3D,
    NumberPlane,
    ParametricFunction,
    RIGHT,
    Sphere,
    ThreeDScene,
    UP,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    Scene,
    Text,
    Transform,
    TracedPath,
    VGroup,
    VMobject,
    ValueTracker,
    always_redraw,
    config,
    linear,
)

config.background_color = "#f6f1e8"

TINTA = "#2c3144"
LILA = "#6d5a8e"
DURAZNO = "#c47b62"
SALVIA = "#3e8f72"
CIELO = "#3d6f99"
ARENA = "#a6843e"
LINEA = "#c4b8a8"


def xy(z, origen=np.zeros(3), escala=1.0):
    return np.asarray(origen, dtype=float) + escala * np.array(
        [np.real(z), np.imag(z), 0.0]
    )


def rotulo(texto, escala=0.38, color=LILA):
    t = Text(texto, font="Lato", color=color, weight="BOLD")
    return t.scale(escala)


def frase(texto, escala=0.34, color=TINTA):
    return Text(texto, font="Lato", color=color).scale(escala)


def camino(zs, color, ancho=4.0, origen=np.zeros(3), escala=1.0):
    mob = VMobject(stroke_color=color, stroke_width=ancho)
    pts = [xy(z, origen, escala) for z in zs]
    if len(pts) < 3:
        pts = pts + [pts[-1]] * (3 - len(pts))
    mob.set_points_as_corners(pts)
    return mob


def muestras(a, b, n=28):
    return [a + (b - a) * i / n for i in range(n)]


def contorno(verts, n=22):
    zs = []
    for i, v in enumerate(verts):
        zs.extend(muestras(v, verts[(i + 1) % len(verts)], n))
    zs.append(verts[0])
    return zs


def cruz(z1, z2, z3, z4):
    return ((z4 - z2) * (z3 - z1)) / ((z4 - z1) * (z3 - z2))


def verificar():
    # cot(θ/2) = sinθ / (1 − cosθ)
    for th in (0.4, 1.0, 2.2):
        a = np.sin(th) / (1 - np.cos(th))
        b = 1 / np.tan(th / 2)
        assert abs(a - b) < 1e-9
    # 1/z es la inversión seguida de la conjugación
    z = 1.4 + 0.7j
    assert abs(np.conj(1 / np.conj(z)) - 1 / z) < 1e-9
    assert abs(abs(z) * abs(1 / np.conj(z)) - 1) < 1e-9
    # razón cruzada real sobre un círculo, y no fuera de él
    c, r = 0.2 + 0.1j, 1.4
    pts = [c + r * np.exp(1j * t) for t in (0.2, 1.3, 2.5, 4.2)]
    real = cruz(*pts)
    assert abs(real.imag) < 1e-8
    fuera = cruz(pts[0], pts[1], pts[2], pts[3] + 0.7j)
    assert abs(fuera.imag) > 1e-2
    # invariante por traslación
    d = 0.4 - 0.3j
    assert abs(cruz(*(p + d for p in pts)) - real) < 1e-8
    # descomposición de una Möbius
    a, b, c_, d_ = 1 + 0.2j, 0.4, -0.5 + 0.1j, 1.2
    z = 0.3 + 0.8j
    directa = (a * z + b) / (c_ * z + d_)
    w = z + d_ / c_
    w = 1 / w
    w = (-(a * d_ - b * c_) / c_**2) * w
    w = w + a / c_
    assert abs(w - directa) < 1e-8


class EsferaDeRiemann(ThreeDScene):
    """Esfera maciza, con meridianos, vista desde arriba y de lado."""

    def construct(self):
        radio = 1.35
        self.set_camera_orientation(phi=58 * DEGREES, theta=-38 * DEGREES, zoom=1.55)
        etiqueta = rotulo("1  ·  ESFERA DE RIEMANN").to_corner(UL, buff=0.32)
        self.add_fixed_in_frame_mobjects(etiqueta)
        self.play(FadeIn(etiqueta))

        esfera = Sphere(
            radius=radio,
            resolution=(42, 28),
            fill_color="#9ec0dc",
            fill_opacity=0.82,
            stroke_width=0,
            checkerboard_colors=False,
        )
        ecuador = Circle(radius=radio, color=CIELO, stroke_width=3.5)

        def curva(func, color=CIELO, ancho=2.4):
            return ParametricFunction(func, t_range=[0, 1], color=color, stroke_width=ancho)

        meridianos = VGroup(
            *[
                curva(
                    lambda t, ph=ph: radio
                    * np.array(
                        [
                            np.sin(np.pi * t) * np.cos(ph),
                            np.sin(np.pi * t) * np.sin(ph),
                            np.cos(np.pi * t),
                        ]
                    )
                )
                for ph in np.linspace(0, np.pi, 4, endpoint=False)
            ]
        )
        paralelos = VGroup(
            *[
                curva(
                    lambda t, z=z: np.array(
                        [
                            np.sqrt(radio**2 - z**2) * np.cos(TAU * t),
                            np.sqrt(radio**2 - z**2) * np.sin(TAU * t),
                            z,
                        ]
                    ),
                    ancho=2.1,
                )
                for z in (0.55 * radio, -0.45 * radio)
            ]
        )
        plano = NumberPlane(
            x_range=[-2.6, 2.6, 1],
            y_range=[-2.6, 2.6, 1],
            background_line_style={"stroke_color": LINEA, "stroke_width": 1.1, "stroke_opacity": 0.9},
            axis_config={"stroke_color": SALVIA, "stroke_width": 2.2},
            faded_line_ratio=0,
        )
        norte = np.array([0.0, 0.0, radio])
        n_dot = Dot3D(point=norte, radius=0.055, color=DURAZNO, resolution=(10, 10))
        et_n = MathTex(r"N=\infty", color=DURAZNO).scale(0.7)
        et_n.move_to(np.array([0.62, -0.48, radio + 0.32]))
        self.add_fixed_orientation_mobjects(et_n)

        self.play(
            FadeIn(plano),
            FadeIn(esfera),
            FadeIn(meridianos),
            FadeIn(paralelos),
            Create(ecuador),
            FadeIn(n_dot, et_n),
            run_time=1.4,
        )

        theta = ValueTracker(2.15)
        azimut = 0.85

        def punto_en_esfera():
            th = theta.get_value()
            return radio * np.array(
                [np.sin(th) * np.cos(azimut), np.sin(th) * np.sin(azimut), np.cos(th)]
            )

        def proyeccion():
            p = punto_en_esfera()
            factor = radio / (radio - p[2])
            return np.array([factor * p[0], factor * p[1], 0.0])

        p_dot = Dot3D(point=punto_en_esfera(), radius=0.05, color=LILA, resolution=(10, 10))
        q_dot = Dot3D(point=proyeccion(), radius=0.05, color=DURAZNO, resolution=(10, 10))
        p_dot.add_updater(lambda m: m.move_to(punto_en_esfera()))
        q_dot.add_updater(lambda m: m.move_to(proyeccion()))
        rayo = always_redraw(
            lambda: Line3D(norte, proyeccion(), color=LILA, thickness=0.016, resolution=8)
        )

        def cartel():
            valor = radio / np.tan(theta.get_value() / 2)
            t = Text(f"|π(P)|   =   {valor:0.2f}", font="Lato", color=TINTA).scale(0.46)
            t.to_edge(DOWN, buff=0.4)
            return t

        lectura = always_redraw(cartel)
        self.add_fixed_in_frame_mobjects(lectura)
        self.add(rayo, p_dot, q_dot)
        self.begin_ambient_camera_rotation(rate=0.16)
        self.play(theta.animate.set_value(0.78), run_time=4.6, rate_func=linear)
        self.wait(0.35)

        casquete = Sphere(
            radius=radio + 0.015,
            resolution=(48, 18),
            v_range=(np.pi - 0.72, np.pi),
            fill_color=DURAZNO,
            fill_opacity=1,
            stroke_width=0,
            checkerboard_colors=False,
        )
        aviso = frase("El casquete del polo es un entorno de ∞.", 0.4, TINTA)
        aviso.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(aviso)
        self.play(FadeOut(lectura), FadeIn(casquete), FadeIn(aviso))
        self.wait(2.6)
        self.stop_ambient_camera_rotation()
        self.wait(0.3)


class MallaMobius(Scene):
    """Una Möbius dobla la malla; el polo se va al infinito."""

    def construct(self):
        etiqueta = rotulo("3  ·  UNA MATRIZ MUEVE EL PLANO").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))

        polo = 0.72 + 0.18j
        a, b, c, d = 1.0 + 0.0j, 0.25 + 0.05j, 1.0 + 0.0j, -polo

        def f(z):
            return (a * z + b) / (c * z + d)

        origen = np.array([-0.2, -0.15, 0.0])
        escala = 1.35

        def trozos(zs):
            grupos = []
            actual_z, actual_w = [], []
            anterior = None
            for z in zs:
                if abs(z - polo) < 0.22:
                    if len(actual_w) > 3:
                        grupos.append((actual_z, actual_w))
                    actual_z, actual_w, anterior = [], [], None
                    continue
                w = f(z)
                if abs(w) > 2.55 or (anterior is not None and abs(w - anterior) > 0.85):
                    if len(actual_w) > 3:
                        grupos.append((actual_z, actual_w))
                    actual_z, actual_w = ([z], [w]) if abs(w) <= 2.55 else ([], [])
                    anterior = w if abs(w) <= 2.55 else None
                    continue
                actual_z.append(z)
                actual_w.append(w)
                anterior = w
            if len(actual_w) > 3:
                grupos.append((actual_z, actual_w))
            return grupos

        lineas = []
        for y in (-1.5, -0.7, 0.15, 0.95, 1.7):
            lineas.append([x + 1j * y for x in np.linspace(-2.15, 2.15, 80)])
        for x in (-1.7, -0.85, 0.05, 0.95, 1.75):
            lineas.append([x + 1j * y for y in np.linspace(-1.7, 1.7, 70)])

        originales = VGroup()
        imagenes = VGroup()
        for zs in lineas:
            for dom, img in trozos(zs):
                originales.add(camino(dom, CIELO, 2.4, origen, escala))
                imagenes.add(camino(img, LILA, 2.6, origen, escala))

        polo_dot = Dot(xy(polo, origen, escala), color=DURAZNO, radius=0.07)
        et_polo = MathTex(r"\mathrm{polo}\; -d/c", color=DURAZNO).scale(0.62)
        et_polo.next_to(polo_dot, LEFT, buff=0.28)
        self.play(FadeIn(originales), FadeIn(polo_dot, et_polo))
        self.wait(0.4)
        self.play(Transform(originales, imagenes), run_time=3.2)
        nota = frase("El polo se va a ∞. Las rectas que lo contienen se abren.", 0.4, TINTA)
        nota.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(nota))
        self.wait(1.5)


class Generadores(Scene):
    """Traslación, dilatación-rotación e inversión generan el grupo."""

    def construct(self):
        etiqueta = rotulo("4.3  ·  TRES MOVIMIENTOS").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([0.15, -0.05, 0.0])
        escala = 1.55
        unidad = Circle(radius=escala, color=LINEA, stroke_width=2).move_to(origen)
        cero = Dot(origen, color=ARENA, radius=0.05)
        self.play(FadeIn(unidad), FadeIn(cero))

        verts = [0.2 + 1.3j, -1.1 + 0.2j, 1.2 - 0.7j]
        pasos = [
            ("Trasladar", lambda z: z + 0.15, SALVIA),
            ("Invertir  1/z", lambda z: 1 / z, DURAZNO),
            ("Girar y estirar", lambda z: (0.95 * np.exp(0.6j)) * z, LILA),
        ]
        forma = camino(contorno(verts), LILA, 4.5, origen, escala)
        self.play(Create(forma))
        leyenda = frase(" ", 0.4)
        self.add(leyenda)
        actual = verts
        for nombre, fn, color in pasos:
            nueva_leyenda = frase(nombre, 0.4, color).to_edge(DOWN, buff=0.38)
            self.play(FadeOut(leyenda), FadeIn(nueva_leyenda), run_time=0.35)
            leyenda = nueva_leyenda
            actual = [fn(z) for z in actual]
            nueva = camino(contorno(actual), color, 4.5, origen, escala)
            self.play(Transform(forma, nueva), run_time=1.6)
            self.wait(0.2)
        cierre = frase("Con estos tres se escribe cualquier Möbius.", 0.38, TINTA)
        cierre.next_to(leyenda, UP, buff=0.18)
        self.play(FadeIn(cierre))
        self.wait(1.3)


class TresPuntos(Scene):
    """Hay una única Möbius que lleva una terna a (∞, 0, 1)."""

    def construct(self):
        etiqueta = rotulo("5  ·  TRES PUNTOS, UNA MÖBIUS").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([-0.4, -0.2, 0.0])
        escala = 1.25
        ejes = VGroup(
            Line(xy(-2.6, origen, escala), xy(3.4, origen, escala), color=LINEA, stroke_width=2),
            Line(xy(-2.2j, origen, escala), xy(2.0j, origen, escala), color=LINEA, stroke_width=2),
        )
        self.play(FadeIn(ejes))

        datos = [
            (-1.55 + 0.85j, r"z_1", DURAZNO, None),
            (0.35 - 0.7j, r"z_2", SALVIA, 0 + 0j),
            (1.7 + 0.55j, r"z_3", CIELO, 1 + 0j),
        ]
        puntos = []
        etiquetas = []
        for z, nombre, color, _ in datos:
            d = Dot(xy(z, origen, escala), color=color, radius=0.08)
            e = MathTex(nombre, color=color).scale(0.62).next_to(d, UP, buff=0.1)
            puntos.append(d)
            etiquetas.append(e)
        self.play(*[FadeIn(m) for m in puntos + etiquetas])
        self.wait(0.4)

        # z2 → 0, z3 → 1, z1 se va y queda el símbolo ∞
        dest2 = xy(0, origen, escala)
        dest3 = xy(1, origen, escala)
        fuga = xy(-1.55 + 0.85j, origen, escala) + np.array([-1.5, 1.7, 0])
        infinito = MathTex(r"\infty", color=DURAZNO).scale(0.9).move_to(np.array([-5.6, 2.5, 0]))
        cero = MathTex("0", color=SALVIA).scale(0.7).next_to(dest2, LEFT, buff=0.16)
        uno = MathTex("1", color=CIELO).scale(0.7).next_to(dest3, RIGHT, buff=0.16)
        self.play(
            puntos[1].animate.move_to(dest2),
            etiquetas[1].animate.set_opacity(0),
            puntos[2].animate.move_to(dest3),
            etiquetas[2].animate.set_opacity(0),
            FadeOut(puntos[0]),
            FadeOut(etiquetas[0]),
            FadeIn(infinito, cero, uno),
            run_time=2.6,
        )
        nota = frase("g(z₁) = ∞,   g(z₂) = 0,   g(z₃) = 1.  Cualquier otra terna se alcanza igual.", 0.4, TINTA)
        nota.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(nota))
        self.wait(1.6)


class CirculoGeneralizado(Scene):
    """Tres puntos dan un círculo; si pasa por el polo, su imagen es una recta."""

    def construct(self):
        etiqueta = rotulo("6.1  ·  CÍRCULOS Y RECTAS").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([-2.6, -0.15, 0.0])
        escala = 1.15
        c, r = 0.15 + 0.05j, 1.35
        angs = (0.4, 2.1, 3.7)
        zs = [c + r * np.exp(1j * a) for a in angs]
        circ = Circle(radius=r * escala, color=LILA, stroke_width=3.2).move_to(xy(c, origen, escala))
        dots = VGroup(*[Dot(xy(z, origen, escala), color=DURAZNO, radius=0.07) for z in zs])
        self.play(FadeIn(dots))
        self.play(Create(circ))
        ley = frase("Tres puntos, un solo círculo generalizado.", 0.36)
        ley.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(ley))
        self.wait(0.8)

        # Círculo por el origen, a la derecha, y su imagen bajo 1/z
        origen2 = np.array([2.85, -0.15, 0.0])
        s = 0.95
        centro = 1.05 + 0.2j
        radio = abs(centro)
        phi = np.angle(-centro)
        cir_der = Circle(radius=radio * s, color=CIELO, stroke_width=3.2).move_to(xy(centro, origen2, s))
        angs_der = [phi + a for a in (1.15, 2.35, 4.0)]
        muestras_z = [centro + radio * np.exp(1j * a) for a in angs_der]
        imagenes_z = [1 / z for z in muestras_z]
        direccion = imagenes_z[1] - imagenes_z[0]
        direccion = direccion / abs(direccion)
        extremo = imagenes_z[0]
        recta = Line(
            xy(extremo - 1.35 * direccion, origen2, s),
            xy(extremo + 1.35 * direccion, origen2, s),
            color=SALVIA,
            stroke_width=3.6,
        )
        viajeros = VGroup(*[Dot(xy(z, origen2, s), color=DURAZNO, radius=0.065) for z in muestras_z])
        cero = Dot(origen2, color=LILA, radius=0.055)
        et0 = MathTex("0", color=LILA).scale(0.55).next_to(cero, LEFT, buff=0.12)
        self.play(FadeOut(ley))
        self.play(Create(cir_der), FadeIn(viajeros), FadeIn(cero, et0))
        self.wait(0.25)
        self.play(
            *[viajeros[i].animate.move_to(xy(imagenes_z[i], origen2, s)) for i in range(3)],
            Create(recta),
            run_time=2.0,
        )
        ley2 = frase("Pasa por el polo: la imagen es una recta.", 0.4, TINTA)
        ley2.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(ley2))
        self.wait(1.4)


class InversionYAngulo(Scene):
    """Un solo dibujo: invertir en el rayo, conjugar, y el ángulo se conserva."""

    def construct(self):
        etiqueta = rotulo("6.2  ·  INVERSIÓN Y  1/z").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([0.15, -0.2, 0.0])
        escala = 1.9
        unidad = Circle(radius=escala, color=CIELO, stroke_width=3.4).move_to(origen)
        real = Line(xy(-1.7, origen, escala), xy(2.05, origen, escala), color=LINEA, stroke_width=2)
        self.play(Create(unidad), FadeIn(real))

        z = 1.38 + 0.42j
        inv = 1 / np.conj(z)
        uno = 1 / z
        dz = Dot(xy(z, origen, escala), color=LILA, radius=0.08)
        dinv = Dot(xy(inv, origen, escala), color=DURAZNO, radius=0.08)
        duno = Dot(xy(uno, origen, escala), color=SALVIA, radius=0.08)
        rayo = DashedLine(origen, xy(z * 1.08, origen, escala), color=LILA, stroke_width=2.4)
        ez = MathTex("z", color=LILA).scale(0.7).next_to(dz, UP, buff=0.1)
        einv = MathTex(r"1/\overline{z}", color=DURAZNO).scale(0.62).next_to(dinv, UP, buff=0.1)
        euno = MathTex("1/z", color=SALVIA).scale(0.7).next_to(duno, DOWN, buff=0.12)
        self.play(FadeIn(dz, ez), Create(rayo))
        self.play(FadeIn(dinv, einv))
        puente = DashedLine(xy(inv, origen, escala), xy(uno, origen, escala), color=SALVIA, stroke_width=2.6)
        self.play(Create(puente), FadeIn(duno, euno))
        nota = frase("La inversión cae en el mismo rayo. Conjugar la baja a 1/z.", 0.4, TINTA)
        nota.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(nota))
        self.wait(1.0)

        deriv = -1 / z**2
        v1 = 0.42 * np.exp(-0.15j)
        v2 = 0.42 * np.exp(0.95j)

        def igual(v):
            w = deriv * v
            return w / abs(w) * abs(v)

        def flecha(a, b, color):
            return Arrow(
                xy(a, origen, escala),
                xy(b, origen, escala),
                buff=0,
                color=color,
                stroke_width=4.5,
                max_tip_length_to_length_ratio=0.18,
            )

        self.play(
            FadeOut(nota),
            FadeIn(flecha(z, z + v1, CIELO), flecha(z, z + v2, DURAZNO)),
        )
        self.play(FadeIn(flecha(uno, uno + igual(v1), CIELO), flecha(uno, uno + igual(v2), DURAZNO)))
        nota2 = frase("Los dos vectores llegan a 1/z con el mismo ángulo.", 0.4, TINTA)
        nota2.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(nota2))
        self.wait(1.6)


class RazonCruzada(Scene):
    """Real exactamente cuando los cuatro puntos son concíclicos, e invariante."""

    def construct(self):
        etiqueta = rotulo("7  ·  LA RAZÓN CRUZADA").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([-1.7, -0.05, 0.0])
        escala = 1.2
        centro, radio = 0.15 + 0.05j, 1.45
        angs = [0.35, 1.45, 2.7, 4.15]
        colores = [DURAZNO, SALVIA, CIELO, LILA]
        nombres = ["z_1", "z_2", "z_3", "z_4"]
        circ = Circle(radius=radio * escala, color=LINEA, stroke_width=2.4).move_to(xy(centro, origen, escala))
        self.play(Create(circ))
        circ.add_updater(
            lambda m: m.move_to(xy(centro + despl.get_value() * (0.55 - 0.35j), origen, escala))
        )

        fuera = ValueTracker(0.0)
        despl = ValueTracker(0.0)

        def posiciones():
            zs = [centro + radio * np.exp(1j * a) for a in angs]
            zs[3] = zs[3] + 1j * fuera.get_value()
            d = despl.get_value() * (0.55 - 0.35j)
            return [z + d for z in zs]

        dots = VGroup()
        tags = VGroup()
        for i in range(4):
            dots.add(Dot(xy(posiciones()[i], origen, escala), color=colores[i], radius=0.07))
            tags.add(MathTex(nombres[i], color=colores[i]).scale(0.5))
        def sync(_):
            zs = posiciones()
            for i in range(4):
                dots[i].move_to(xy(zs[i], origen, escala))
                tags[i].next_to(dots[i], UP if i != 1 else DOWN, buff=0.08)
        dots.add_updater(sync)
        self.add(dots, tags)
        sync(None)

        real = DecimalNumber(0, num_decimal_places=2, color=TINTA, font_size=36, include_sign=True)
        imag = DecimalNumber(0, num_decimal_places=2, color=TINTA, font_size=36, include_sign=True)
        caja = VGroup(
            MathTex(r"(z_1,z_2;z_3,z_4)=", color=TINTA).scale(0.7),
            real,
            MathTex("+", color=TINTA).scale(0.7),
            imag,
            MathTex(r"\,i", color=TINTA).scale(0.7),
        ).arrange(RIGHT, buff=0.1)
        caja.to_edge(DOWN, buff=0.42)

        def refrescar(_):
            valor = cruz(*posiciones())
            real.set_value(valor.real)
            imag.set_value(valor.imag)

        dots.add_updater(refrescar)
        refrescar(None)
        self.play(FadeIn(caja))
        self.wait(0.6)
        estado = frase("Sobre el círculo, la razón es real.", 0.36, SALVIA)
        estado.to_edge(UP, buff=0.95).shift(RIGHT * 1.2)
        self.play(FadeIn(estado))
        self.wait(0.5)
        self.play(FadeOut(estado), fuera.animate.set_value(0.85), run_time=2.0)
        estado2 = frase("Sale del círculo: aparece una parte imaginaria.", 0.36, DURAZNO)
        estado2.to_edge(UP, buff=0.95).shift(RIGHT * 0.6)
        self.play(FadeIn(estado2))
        self.wait(0.7)
        self.play(FadeOut(estado2), fuera.animate.set_value(0.0), run_time=1.4)
        self.play(despl.animate.set_value(1.0), run_time=2.0)
        estado3 = frase("Una traslación mueve los puntos. El número queda.", 0.36, LILA)
        estado3.to_edge(UP, buff=0.95).shift(RIGHT * 0.5)
        self.play(FadeIn(estado3))
        self.wait(1.4)


class ClasificacionPorTraza(Scene):
    """Girar, trasladar, estirar o espiral, según la traza."""

    def construct(self):
        etiqueta = rotulo("8  ·  CUATRO MOVIMIENTOS").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([-1.15, -0.1, 0.0])
        escala = 1.35
        ejes = VGroup(
            Line(xy(-2.4, origen, escala), xy(2.5, origen, escala), color=LINEA, stroke_width=2),
            Line(xy(-1.8j, origen, escala), xy(1.8j, origen, escala), color=LINEA, stroke_width=2),
        )
        fijo = Dot(origen, color=ARENA, radius=0.06)
        et_fijo = MathTex("0", color=ARENA).scale(0.55).next_to(fijo, LEFT, buff=0.18)
        self.play(FadeIn(ejes))

        semillas = [
            x + 1j * y
            for x in np.linspace(-1.05, 1.05, 5)
            for y in np.linspace(-0.85, 0.85, 4)
            if abs(x + 1j * y) > 0.2
        ]
        guia = 0.75 + 0.45j
        semillas.append(guia)
        dots = VGroup(
            *[
                Dot(xy(z, origen, escala), color=DURAZNO if z == guia else LILA, radius=0.075 if z == guia else 0.05)
                for z in semillas
            ]
        )
        self.play(FadeIn(dots))

        def ir(fn, nombre, detalle, color, segundos=2.3, mostrar_cero=True):
            tracker = ValueTracker(0)
            texto = VGroup(
                frase(nombre, 0.46, color),
                frase(detalle, 0.32, TINTA),
            ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            texto.to_edge(RIGHT, buff=0.4).shift(UP * 0.3)

            def upd(mob):
                t = tracker.get_value()
                for dot, z in zip(mob, semillas):
                    w = fn(z, t)
                    dot.move_to(xy(w, origen, escala))

            dots.add_updater(upd)
            traza = TracedPath(dots[-1].get_center, stroke_color=color, stroke_width=3.4)
            self.add(traza)
            extras = [FadeIn(texto)]
            if mostrar_cero:
                extras += [FadeIn(fijo), FadeIn(et_fijo)]
            self.play(*extras)
            self.play(tracker.animate.set_value(1), run_time=segundos, rate_func=linear)
            dots.remove_updater(upd)
            self.wait(0.85)
            self.play(
                FadeOut(texto),
                FadeOut(traza),
                *[d.animate.move_to(xy(z0, origen, escala)) for d, z0 in zip(dots, semillas)],
                run_time=0.55,
            )
            self.remove(traza)

        ir(lambda z, t: z + t * (1.45 - 0.2j), "Parabólica", "un solo fijo: ∞", DURAZNO, mostrar_cero=False)
        self.play(FadeIn(fijo), FadeIn(et_fijo))
        ir(lambda z, t: z * np.exp(1j * t * 2.1), "Elíptica", "gira: traza entre −2 y 2", CIELO)
        ir(lambda z, t: z * (1 + 1.15 * t), "Hiperbólica", "estiramiento real, k > 0", SALVIA)
        ir(
            lambda z, t: z * np.exp(t * (0.62 + 1.35j)),
            "Loxodrómica",
            "espiral: la traza no es real",
            LILA,
        )
        self.wait(0.6)


class InversaYVuelta(Scene):
    """Una figura se va con f y regresa con f⁻¹."""

    def construct(self):
        etiqueta = rotulo("3  ·  LA INVERSA DEVUELVE").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([0.15, -0.1, 0.0])
        escala = 1.65

        giro = 1.25 * np.exp(0.95j)
        paso = 0.85 - 0.35j

        def f(z):
            return giro * z + paso

        def finv(z):
            return (z - paso) / giro

        verts = [0.15 + 0.85j, -0.7 + 0.15j, 0.35 - 0.65j]
        imagen = [f(z) for z in verts]
        figura = camino(contorno(verts), LILA, 3.6, origen, escala)
        self.play(Create(figura))
        nombre = MathTex("f", color=DURAZNO).scale(0.9).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(nombre))
        self.play(Transform(figura, camino(contorno(imagen), LILA, 3.6, origen, escala)), run_time=2.2)
        self.wait(0.8)
        self.play(FadeOut(nombre))
        nombre = MathTex("f^{-1}", color=SALVIA).scale(0.9).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(nombre))
        self.play(Transform(figura, camino(contorno([finv(z) for z in imagen]), LILA, 3.6, origen, escala)), run_time=2.2)
        nota = frase("Volvió al mismo triángulo.", 0.4, TINTA).next_to(nombre, UP, buff=0.2)
        self.play(FadeIn(nota))
        self.wait(1.2)


class ALoMasDos(Scene):
    """Dos puntos quietos; el tercero tiene que moverse."""

    def construct(self):
        etiqueta = rotulo("5  ·  A LO MÁS DOS FIJOS").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([-0.45, -0.45, 0.0])
        escala = 1.3
        eje = Line(xy(-1.3, origen, escala), xy(2.3, origen, escala), color=LINEA, stroke_width=2)
        self.play(Create(eje))

        fijos = [0j, 1 + 0j]
        libres = [-0.55 + 0.7j, 0.5 + 0.45j, -0.3 - 0.55j]
        marcas = VGroup(
            *[Dot(xy(z, origen, escala), color=DURAZNO, radius=0.09) for z in fijos]
        )
        etiquetas = VGroup(
            MathTex("0", color=DURAZNO).scale(0.7).next_to(marcas[0], DOWN, buff=0.14),
            MathTex("1", color=DURAZNO).scale(0.7).next_to(marcas[1], DOWN, buff=0.14),
        )
        puntos = VGroup(*[Dot(xy(z, origen, escala), color=LILA, radius=0.07) for z in libres])
        puntos[0].set_color(CIELO).scale(1.15)
        self.play(FadeIn(marcas, etiquetas, puntos))

        def mapa(z, t):
            k = np.exp(1.05 * t)
            return k * z / ((k - 1) * z + 1)

        tracker = ValueTracker(0)
        traza = TracedPath(puntos[0].get_center, stroke_color=CIELO, stroke_width=3.4)

        def upd(mob):
            t = tracker.get_value()
            for dot, z in zip(mob, libres):
                dot.move_to(xy(mapa(z, t), origen, escala))

        puntos.add_updater(upd)
        self.add(traza)
        ley = frase("0 y 1 no se mueven. El otro punto sí.", 0.4, TINTA)
        ley.to_edge(DOWN, buff=0.38)
        self.play(FadeIn(ley))
        self.play(tracker.animate.set_value(1), run_time=3.2, rate_func=linear)
        puntos.remove_updater(upd)
        self.wait(1.3)


class MismaTraza(Scene):
    """Dos traslaciones con la misma traza: la misma parabólica, en otra dirección."""

    def construct(self):
        etiqueta = rotulo("8  ·  LA MISMA TRAZA").to_corner(UL, buff=0.32)
        self.play(FadeIn(etiqueta))
        origen = np.array([-0.7, -0.15, 0.0])
        escala = 1.45
        semillas = [0.4 + 0.25j, -0.25 + 0.5j, -0.5 - 0.15j, 0.2 - 0.45j, 0.65 + 0.6j]
        dots = VGroup(
            *[
                Dot(xy(z, origen, escala), color=DURAZNO if z == semillas[-1] else LILA, radius=0.075 if z == semillas[-1] else 0.05)
                for z in semillas
            ]
        )
        self.play(FadeIn(dots))

        def correr(direccion, nombre, color):
            tracker = ValueTracker(0)
            texto = frase(nombre, 0.46, color).to_edge(RIGHT, buff=0.45).shift(UP * 0.4)

            def upd(mob):
                t = tracker.get_value()
                for dot, z in zip(mob, semillas):
                    dot.move_to(xy(z + t * direccion, origen, escala))

            dots.add_updater(upd)
            traza = TracedPath(dots[-1].get_center, stroke_color=color, stroke_width=3.4)
            self.add(traza)
            self.play(FadeIn(texto))
            self.play(tracker.animate.set_value(1), run_time=2.2, rate_func=linear)
            dots.remove_updater(upd)
            self.wait(0.7)
            self.play(
                FadeOut(texto),
                FadeOut(traza),
                *[d.animate.move_to(xy(z, origen, escala)) for d, z in zip(dots, semillas)],
                run_time=0.5,
            )
            self.remove(traza)

        correr(1.4 + 0j, "z  ↦  z + 1", DURAZNO)
        correr(1.4 * np.exp(1.05j), "la misma, en otra dirección", SALVIA)
        cierre = frase("Las dos tienen tr² = 4. Son conjugadas.", 0.4, TINTA)
        cierre.to_edge(DOWN, buff=0.38)
        self.play(FadeIn(cierre))
        self.wait(1.3)
