from manim import *
from scipy.special import erf


LINE = r"""u(x, t) =
\begin{cases} \displaystyle
\dfrac{10}{\sqrt{\pi t}} \int \limits _0^1 (1-s) \mathrm{e} ^ { -100\frac{(x-s)^2}{t} } \mathrm{d} s,\ t > 0,\\
\varphi (x),\ t = 0.
\end{cases}"""


def step(x):
    """
    Ступенчатая функция Хевисайда
    """
    if x < 0:
        return 0
    return 1


def temperature(x, t, a):
    """
    Решение задачи.
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    elif t == 0:
        return (1-x) * (step(x) - step(x-1))
    g = (1-x) * 0.5 * (erf(x/(2*a*np.sqrt(t))) - erf((x-1)/(2*a*np.sqrt(t))))
    h = -a * np.sqrt(t/np.pi) * (np.exp(- x**2 / (4 * a**2 * t)
                                        ) - np.exp(- (x-1)**2 / (4 * a**2 * t)))
    return g + h


def moving_label(t, car, f_s=24):
    """
    Кусок движущегося текста
    """
    return Tex("t={t}", font_size=f_s).next_to(car, UR)


class plane_animation(Scene):
    def construct(self):
        """
        Пространственно-временная анимация полученного решения.
        """
        # Константы
        A = 0.5 / np.sqrt(5)
        END_OF_TIME = 5
        FONT_SIZE = 24
        Y_LENGTH, X_LENGTH = 8.4, 14

        T = ValueTracker(1e-5)

        axes = Axes(y_length=Y_LENGTH, x_length=X_LENGTH,
                    tips=0).add_coordinates()

        nl = NumberLine(
            font_size=FONT_SIZE,
            x_range=[0, END_OF_TIME, 1],
            numbers_to_include=[0, END_OF_TIME],
        )

        time_stamp = Dot(color=YELLOW).add_updater(
            lambda mob: mob.move_to(nl.number_to_point(T.get_value())),
        ).update()

        function_label = MathTex(LINE, font_size=FONT_SIZE)

        label = MathTex(r"t", font_size=FONT_SIZE).next_to(time_stamp, UP)
        label.add_updater(
            lambda mob: mob.next_to(time_stamp, UP),
        ).update()

        graph = FunctionGraph(lambda x: temperature(
            x, T.get_value(), A), x_range=[-7, 7], color=RED)
        graph.add_updater(
            lambda func: func.become(
                FunctionGraph(lambda x: temperature(
                    x, T.get_value(), A), x_range=[-7, 7], color=RED)
            )
        )

        plot = VGroup(axes, graph)

        nl.to_corner(UR)
        time_stamp.move_to(nl.number_to_point(0))
        label.next_to(time_stamp, UP)
        function_label.to_corner(UL)

        self.add(function_label, label, nl)
        self.add(time_stamp, plot)
        self.wait(1)
        self.play(T.animate.set_value(END_OF_TIME), run_time=END_OF_TIME, rate_func=rate_functions.linear)
