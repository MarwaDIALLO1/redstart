import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci
    from scipy.linalg import solve_continuous_are 
    from scipy.integrate import solve_ivp
    import matplotlib as mpl
    from numpy.linalg import inv
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return (
        FFMpegWriter,
        FuncAnimation,
        inv,
        mpl,
        np,
        plt,
        scipy,
        solve_continuous_are,
        solve_ivp,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    To find the possible equilibria of the system with constant inputs f and ϕ, we must analyze the differential equations that govern the dynamics of the booster.

    We already have the Equations of Motion

    **Center of mass:**
    - \( M\ddot{x} = -f \sin(\theta + \varphi) \)
    - \( M\ddot{y} = f \cos(\theta + \varphi) - Mg \)

    **Tilt (angle):**
    - \( J\ddot{\theta} = -\ell \sin(\varphi)f \)

    An equilibrium state corresponds to a state where all time derivatives are zero, that is:

    - \( \ddot{x} = 0 \)
    - \( \ddot{y} = 0 \)
    - \( \ddot{\theta} = 0 \)

    1. From the tilt equation:  
       \( J\ddot{\theta} = -\ell \sin(\varphi)f = 0 \Rightarrow \sin(\varphi) = 0 \Rightarrow \varphi = 0 \) or \( \pi \).  
       But since \( |\varphi| < \frac{\pi}{2} \), only \( \varphi = 0 \) is valid.

    2. From the vertical equation (y):  
       \( M\ddot{y} = f \cos(\theta + \varphi) - Mg = 0 \Rightarrow f \cos(\theta + \varphi) = Mg \).  
       Since \( \varphi = 0 \), this gives:  
       \( f \cos(\theta) = Mg \)

    3. From the horizontal equation (x):  
       \( M\ddot{x} = -f \sin(\theta + \varphi) = 0 \Rightarrow \sin(\theta + \varphi) = 0 \Rightarrow \theta + \varphi = 0 \mod \pi \).  
       With \( \varphi = 0 \), we get:  
       \( \theta = 0 \mod \pi \).  
       But since \( |\theta| < \frac{\pi}{2} \), only \( \theta = 0 \) is valid.


    **The only possible equilibrium states under the constraints \( |\theta| < \frac{\pi}{2} \), \( |\varphi| < \frac{\pi}{2} \), and \( f > 0 \) are:**

    - **\( \theta = 0 \)**
    - **\( \varphi = 0 \)**
    - **\( f = Mg \)**
    This is available for all the values of x and y 
    Therefore, the only possible equilibrium is when the booster is perfectly vertical (\( \theta = 0 \)), the force is aligned with its axis (\( \varphi = 0 \)), and the thrust exactly balances the weight of the booster (\( f = Mg \)).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **1. Definition of Perturbations**

    We introduce small variations around the equilibrium:

    $$
    \begin{aligned}
    x &= x_{\text{eq}} + \Delta x, \\
    y &= y_{\text{eq}} + \Delta y, \\
    \theta &= \theta_{\text{eq}} + \Delta \theta, \\
    \phi &= \phi_{\text{eq}} + \Delta \phi, \\
    f &= f_{\text{eq}} + \Delta f.
    \end{aligned}
    $$

    The perturbed accelerations are:

    $$
    \Delta \ddot{x} = \ddot{x}, \quad \Delta \ddot{y} = \ddot{y}, \quad \Delta \ddot{\theta} = \ddot{\theta}.
    $$

    **2. Linearization of the Equations**

    We approximate the equations using a first-order Taylor expansion.


    **Linearization of Trigonometric Terms**

    Using a first-order Taylor expansion:

    $$
    \sin(\theta + \phi) \approx \sin(\theta_{\text{eq}} + \phi_{\text{eq}}) 
    + \cos(\theta_{\text{eq}} + \phi_{\text{eq}})(\Delta \theta + \Delta \phi)
    $$

    $$
    \cos(\theta + \phi) \approx \cos(\theta_{\text{eq}} + \phi_{\text{eq}}) 
    - \sin(\theta_{\text{eq}} + \phi_{\text{eq}})(\Delta \theta + \Delta \phi)
    $$

    $$
    \sin(\phi) \approx \sin(\phi_{\text{eq}}) + \cos(\phi_{\text{eq}})\Delta \phi
    $$

    **Equation (1): Horizontal Motion ($x$)**

    $$
    M \Delta\ddot{x} = - (f_{\text{eq}} + \Delta f) \left[ \sin(\theta_{\text{eq}} + \phi_{\text{eq}}) + \cos(\theta_{\text{eq}} + \phi_{\text{eq}})(\Delta\theta + \Delta\phi) \right]
    $$

    Neglecting second-order terms such as \( \Delta f \cdot \Delta\theta \), we obtain:

    $$
    M \, \Delta \ddot{x} \approx 
    - f_{\text{eq}} \cos(\theta_{\text{eq}} + \phi_{\text{eq}})(\Delta \theta + \Delta \phi)
    - \sin(\theta_{\text{eq}} + \phi_{\text{eq}}) \, \Delta f
    $$

    **Equation (2): Vertical Motion ($y$)**

    $$
    M \Delta\ddot{y} = (f_{\text{eq}} + \Delta f) \left[ \cos(\theta_{\text{eq}} + \phi_{\text{eq}}) - \sin(\theta_{\text{eq}} + \phi_{\text{eq}})(\Delta\theta + \Delta\phi) \right] - Mg
    $$

    $$
    M \, \Delta \ddot{y} \approx 
    - f_{\text{eq}} \sin(\theta_{\text{eq}} + \phi_{\text{eq}})(\Delta \theta + \Delta \phi)
    + \cos(\theta_{\text{eq}} + \phi_{\text{eq}}) \, \Delta f
    $$

    **Equation (3): Rotation ($\theta$)**

    $$
    J \Delta \ddot{\theta} = - \ell (f_{\text{eq}} + \Delta f) \left[ \sin(\phi_{\text{eq}}) + \cos(\phi_{\text{eq}}) \Delta \phi \right]
    $$

    $$
    J \, \Delta \ddot{\theta} \approx 
    - \ell f_{\text{eq}} \cos(\phi_{\text{eq}}) \, \Delta \phi 
    - \ell \sin(\phi_{\text{eq}}) \, \Delta f
    $$




    **3. Special Case: Standard Equilibrium**

    If the equilibrium is:

    $$
    \theta_{\text{eq}} = 0, \quad \phi_{\text{eq}} = 0, \quad f_{\text{eq}} = Mg,
    $$

    then the equations simplify to:

    $$
    \begin{cases}
    \Delta \ddot{x} = -g (\Delta \theta + \Delta \phi), \\
    \Delta \ddot{y} = \dfrac{\Delta f}{M}, \\
    \Delta \ddot{\theta} = -\dfrac{M \ell g}{J} \Delta \phi.
    \end{cases}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We want to express the linearized equations in the standard form of a linear dynamic system, namely:  

    \[
    \dot{x} = A x + B u
    \]

    where:  
    - \( x \) is the state vector,  
    - \( A \) is the state matrix,  
    - \( B \) is the input matrix,  
    - \( u \) is the input vector.  



    First, we introduce the error variables :  
    \( \Delta x, \Delta \dot{x}, \Delta y, \Delta \dot{y}, \Delta \theta, \Delta \dot{\theta} \).  
    The state vector is therefore:  

    \[
    x = 
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    \]

    **Inputs:**  
    The perturbations of the inputs are:  
    \( \Delta f \) and \( \Delta \varphi \).  
    The input vector is:  

    \[
    u = 
    \begin{bmatrix}
    \Delta f \\
    \Delta \varphi
    \end{bmatrix}
    \]



    **Linearized differential equations:**  
    From the previous question :

    \[
    \begin{aligned}
    \Delta \ddot{x} &= -g(\Delta \theta + \Delta \varphi), \\
    \Delta \ddot{y} &= \frac{1}{M} \Delta f, \\
    \Delta \ddot{\theta} &= -\frac{M g \ell}{J} \Delta \varphi.
    \end{aligned}
    \]



    **Second-order canonical form:**  
    For each error variable, we introduce an auxiliary variable corresponding to its first derivative. For example:  
    \( \Delta \dot{x} = \Delta v_x \), \( \Delta \dot{y} = \Delta v_y \), \( \Delta \dot{\theta} = \Delta \omega \).  
    The equations then become:

    \[
    \begin{aligned}
    \Delta \dot{x} &= \Delta v_x \\
    \Delta \dot{v}_x &= -g(\Delta \theta + \Delta \varphi) \\
    \Delta \dot{y} &= \Delta v_y \\
    \Delta \dot{v}_y &= \frac{1}{M} \Delta f \\
    \Delta \dot{\theta} &= \Delta \omega \\
    \Delta \dot{\omega} &= -\frac{M g \cdot \ell}{J} \Delta \varphi
    \end{aligned}
    \]


    **State matrix A:**  
    Matrix A is constructed by aligning the coefficients of the state variables in the equations above. It has the form:

    \[
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \]


    **Input matrix B:**  
    Matrix B contains the coefficients of the inputs \( \Delta f \) and \( \Delta \varphi \). It has the form:

    \[
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{M g \ell}{J}
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell
def _(A):
    from scipy import linalg
    eigenvalues, _ = linalg.eig(A)
    eigenvalues
    return


@app.cell
def _(mo):
    mo.md(r"""All eigenvalues are zero. This means the system is not asymptotically stable.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The controllability matrix \( \mathcal{C} \) is computed as:

    \[
    \mathcal{C} = \begin{bmatrix}
    B & AB & A^2B & A^3B & A^4B & A^5B
    \end{bmatrix}
    \]

    Where:

    - \( B \in \mathbb{R}^{6 \times 2} \)
    - \( \mathcal{C} \in \mathbb{R}^{6 \times 12} \)

    Rank Analysis

    Using the python computation with the values (\( g = 1, M = 1, l = 1, J = 1 \)), the rank of the controllability matrix is:

    \[
    \text{rank}(\mathcal{C}) = 6
    \]


    Since:

    \[
    \text{rank}(\mathcal{C}) = n = 6
    \]

    *The system is controllable.*
    """
    )
    return


@app.cell
def _(J, M, g, l, np):
    from numpy.linalg import matrix_power

    def KCM(A, B):
        n = np.shape(A)[0]
        return np.column_stack([matrix_power(A, k) @ B for k in range(n)])


    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])
    B = np.array([
        [0, 0],
        [0, -g],
        [0, 0],
        [1/M, 0],
        [0, 0],
        [0, -l * g * M / J]
    ])


    # Calculate the controllability matrix
    C = KCM(A, B)
    print("Controllability matrix shape:", C.shape)
    print("Rank of controllability matrix:", np.linalg.matrix_rank(C))
    return A, KCM


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    \[
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0\\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}
    \]

    \[
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    0 & -\frac{ M g \ell}{J}
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell
def _(J, KCM, M, g, l, np):
    A1 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    # Input matrix B
    B1 = np.array([
        [0, 0],
        [0, -g],
        [0, 0],
        [0, -l * g * M / J]
    ])

    # Calculate the controllability matrix
    C1 = KCM(A1, B1)
    print("Controllability matrix shape:", C1.shape)
    print("Rank of controllability matrix:", np.linalg.matrix_rank(C1))
    return


app._unparsable_cell(
    r"""
    Thus, the new system is controllable.
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Special Case: Free Fall ($\phi(t) = 0$, $f = 0$)**
    In this case:
    $\Delta \phi = 0$,
    $\Delta f = -Mg$ (since $f_{\text{eq}} = Mg$ and $f = 0$).

    The equations become:

    $$
    \begin{aligned}
    \Delta \ddot{x} &= -g \Delta \theta, \\
    \Delta \ddot{y} &= -g, \\
    \Delta \ddot{\theta} &= 0.
    \end{aligned}
    $$


    **Initial Conditions**

    - $x(0) = 0$, $\dot{x}(0) = 0$
    - $y(0) = 10$, $\dot{y}(0) = 0$
    - $\theta(0) = \frac{\pi}{4}$, $\dot{\theta}(0) = 0$


    **Analytical Solutions**

    **Angle $\theta(t)$**

    $$
    \ddot{\theta} = 0 \Rightarrow \dot{\theta} = \dot{\theta}(0) =0 \Rightarrow\theta(t) = \theta(0) = \frac{\pi}{4}
    $$

    *$\theta(t)$ remains constant.*

    **Vertical position $y(t)$**

    $$
    \ddot{y} = -g \Rightarrow \dot{y}(t) = -g t ,\quad y(t) = -\frac{1}{2} g t^2 + y_0
    $$
    """
    )
    return


@app.cell
def _(g, np, plt):
    theta0 = np.pi / 4  
    y0 = 10
    t = np.linspace(0, 2, 500)


    theta = theta0 * np.ones_like(t)

    y = -0.5 * g * t**2  + y0

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, y, label=r'$y(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical Position $y(t)$')
    plt.title('Vertical Motion: $y(t)$')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, theta, 'r', label=r'$\theta(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle $\theta(t)$')
    plt.title(r'Orientation $\theta(t)$')
    plt.grid(True)
    plt.legend()
    return


app._unparsable_cell(
    r"""
    **Left Plot: Vertical Motion \( y(t) \)**

    - The plot shows a parabolic decrease in the vertical position \( y(t) \).

    - This indicates that the system is falling freely with no vertical thrust or corrective force applied.



    **Right Plot: Orientation \( \theta(t) \)**

    - The orientation \( \theta(t) \) remains almost constant at approximately \( 0.78 \) radians throughout the simulation.

    - The system starts close to its equilibrium angle \( \theta_{\text{eq}} \),



    **Conclusion**

    - The object exhibits free-fall vertical motion while maintaining a stable orientation.
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We are given the control law:

    \[
    \Delta\phi(t) = -K \cdot 
    \begin{bmatrix}
    \Delta x(t) \\
    \dot{\Delta x}(t) \\
    \Delta\theta(t) \\
    \dot{\Delta\theta}(t)
    \end{bmatrix}, \quad K = 
    \begin{bmatrix}
    0 \\
    0 \\
    k_3 \\
    k_4
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 1}
    \]

    We aim to determine \(k_3\) and \(k_4\) such that \( \Delta\theta(0) = \frac{\pi}{4} \), \(\dot{\Delta\theta}(0) = 0\), and \(\Delta x(0) = \dot{\Delta x}(0) = 0\).

    The physical system is governed by the second-order dynamics:

    \[
    \ddot{\Delta\theta}(t) = -\frac{l g M}{J} \Delta\phi(t)
    \]

    By injecting the control law:

    \[
    \Delta\phi(t) = -k_3 \Delta\theta(t) - k_4 \dot{\Delta\theta}(t)
    \Rightarrow 
    \ddot{\Delta\theta}(t) = \frac{l g M}{J} (k_3 \Delta\theta + k_4 \dot{\Delta\theta})
    \]

    This yields a standard second-order linear differential equation:

    \[
    \ddot{\Delta\theta}(t) + a_1 \dot{\Delta\theta}(t) + a_0 \Delta\theta(t) = 0
    \]

    With coefficients:

    \[
    a_1 = -\frac{l g M}{J} k_4, \quad a_0 = -\frac{l g M}{J} k_3
    \]

    To design a well-behaved second-order system, we match the form:

    \[
    \ddot{\theta} + 2\zeta\omega_n \dot{\theta} + \omega_n^2 \theta = 0
    \]

    **Desired Behavior:**
    - Settling time \( T_s \leq 20 \) seconds  
    - Therefore, choose \( \omega_n = 0.3 \) (so that \( T_s \approx \frac{4}{\zeta \omega_n} \leq 20 \))
    - Damping ratio \( \zeta = 0.8 \)
 
    $$
    a_1 = 2\zeta\omega_n = 0.48
    $$

    $$
    a_0 = \omega_n^2 = 0.09
    $$

    **System Parameters:**
    - \( l = 1 \)
    - \( g = 1 \)
    - \( M = 1 \)
    - \( J = 0.33 \)
    - So, \( \frac{l g M}{J} = \frac{1}{0.33} \approx 3.03 \)

    **Compute gains:**

    $$
    k_4 = -\frac{J}{l g M} a_1 \approx -0.33 \cdot 0.48 = -0.1584
    $$

    $$
    k_3 = -\frac{J}{l g M} a_0 \approx -0.33 \cdot 0.09 = -0.0297
    $$


    We now simulate the system to verify constraints:

    - \( |\Delta\theta(t)| < \frac{\pi}{2} \)
    - \( |\Delta\phi(t)| < \frac{\pi}{2} \)
    - \( \Delta\theta(t) \rightarrow 0 \) in about 20 seconds
    """
    )
    return


@app.cell
def _(A_2, B_2, np):
    from numpy.linalg import eig

    K = np.array([[0, 0, -0.0297, -0.1584]])  # (1 x 4)

    # Calcul de la matrice fermée
    A_cl = A_2 - B_2 @ K  # (4x1) x (1x4) = (4x4)

    # Valeurs propres
    eigenvalue, _ = eig(A_cl)

    # Vérification de la stabilité
    is_stable = all(np.real(eigenvalue) < 0)

    # Affichage
    print("Eigenvalues:", eigenvalue)
    print("Stable:", is_stable)
    return


@app.cell
def _(J, M, g, l, np, plt, solve_ivp):
    D = l * g * M / J  # ≈ 3.03

    # Gains from design
    k3 = -0.0297
    k4 = -0.1584

    # Dynamics: [thetaa, theta_dot]
    def dynamics(t, y):
        thetaa, theta_dot = y
        phi = -k3 * thetaa - k4 * theta_dot
        theta_ddot = D * (k3 * thetaa + k4 * theta_dot)
        return [theta_dot, theta_ddot]

    # Initial conditions
    thetaa0 = np.pi / 4
    theta_dot0 = 0
    y00 = [thetaa0, theta_dot0]

    # Time span
    t_evalu = np.linspace(0, 30, 1000)
    so = solve_ivp(dynamics, [0, 30], y00, t_evalu=t_evalu)

    # Compute phi over time
    thetaa = so.y[0]
    theta_dot = so.y[1]
    phi = -k3 * thetaa - k4 * theta_dot

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(so.t, thetaa, label=r'$\Delta \theta_a(t)$')
    plt.plot(so.t, phi, label=r'$\Delta\phi(t)$')
    plt.axhline(np.pi/2, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(-np.pi/2, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.title('Response of the system with designed $k_3$ and $k_4$')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The reduced model of the system is given by: 

    \[
    \dot{x} = A x + B \Delta \phi
    \]

    where:

    \[
    x =
    \begin{bmatrix}
    \Delta x, \Delta \dot{x}, \Delta \theta, \Delta \dot{\theta}
    \end{bmatrix}^T
    \]

    \[
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}
    ,\quad
    B =
    \begin{bmatrix}
    0 \\
    - g \\
    0 \\
    - \frac{Mg\ell}{J}
    \end{bmatrix}
    \]

    We want:

    - Asymptotic stability → all closed-loop poles must have a negative real part.
    - Short settling time → choose poles far to the left in the complex plane, ideally with fast dynamics but without strong oscillations (avoid complex poles with high imaginary parts).
    - Reduction of Δx(t) → 0 in less than 20 seconds 


    **Pole Selection**

    We choose negative real poles to insure the asymptotic stability :

    \[
    p_1 = -0.5,\quad p_2 = -1.0,\quad p_3 = -1.5,\quad p_4 = -2.0
    \]

    These poles ensure relatively fast convergence to zero (settling time less than 20s).


    We can obtain:

    \[
    K_{pp} = 
    \begin{bmatrix}
    0.5 \\
    2.0833 \\
    -3.0833 \\
    -2.3611
    \end{bmatrix}
    \]

    So:

    \[
    \Delta \phi(t) = -0.5 \cdot \Delta x(t) - 2.0833 \cdot \Delta \dot{x}(t) + 3.0833 \cdot \Delta \theta(t) + 2.3611 \cdot \Delta \dot{\theta}(t)
    \]

    Once the gain \( K_{pp} \) is found, we simulate the closed-loop system:

    \[
    \dot{x} = (A - B K_{pp}) x
    \]

    The simulation provided by the code below verify that:

    - The states \( \Delta x(t), \Delta \theta(t) \) converge to 0,
    - The settling time of \( \Delta x(t) \) is well under 20 seconds,
    - The control \( \Delta \phi(t) \) remains bounded.


    """
    )
    return


@app.cell
def _(J, M, g, l, np):
    from scipy.signal import place_poles
    # Matrices du système
    A_2 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_2 = np.array([
        [0],
        [-g],
        [0],
        [-M*g*l/J]
    ])

    # Pôles choisies
    poles = [-0.5, -1.0, -1.5, -2.0]

    # Calcul du gain K
    K_pp = place_poles(A_2, B_2, poles).gain_matrix.flatten()

    print("Matrice de gain K_pp :")
    print(K_pp)
    return A_2, B_2, K_pp


@app.cell
def _(A_2, B_2, K_pp, np, plt, solve_ivp):
    # Matrice du système en boucle fermée
    A_2_cl = A_2 - B_2 @ K_pp[np.newaxis, :]  # Shape correcte de K_pp

    # Fonction d'équation différentielle
    def closed_loop_system(t, x):
        return A_2_cl @ x

    # Conditions initiales
    x0 = np.array([1.0, 0.0, 0.5, 0.0])  # [Δx, Δẋ, Δθ, Δθ̇]

    # Intervalle de temps
    t_span = [0, 30]  # Simuler sur 30 secondes
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Résolution de l'EDO
    sol = solve_ivp(closed_loop_system, t_span, x0, t_eval=t_eval)

    # Extraction des résultats
    e = sol.t
    x = sol.y

    #t_2Calcul de la commande Δϕ(e) = -K_pp ⋅ x(e)
    u = -K_pp @ x

    # Tracé des états
    plt.figure(figsize=(14, 10))

    # Δx(e)
    plt.subplot(2, 2, 1)
    plt.plot(e, x[0], label=r'$\Delta x(e)$')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(r'$\Delta x(e)$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\Delta x$')
    plt.grid(True)


    # Δθ(e)
    plt.subplot(2, 2, 3)
    plt.plot(e, x[2], 'r', label=r'$\Delta \theta(t)$')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(r'$\Delta \theta(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\Delta \theta$')
    plt.grid(True)

    # Tracé de la commande Δϕ(t)
    plt.figure(figsize=(10, 4))
    plt.plot(e, u, label=r'$\Delta \phi(t)$')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(r'$\Delta \phi(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\Delta \phi$')
    plt.grid(True)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    By the  LQR method we aim to minimize a quadratic cost function of the form:

    \[
    J = \int_0^{\infty} \left( x^T Q x + \Delta \phi^T R \Delta \phi \right) dt
    \]

    where:

    - \( x = \begin{bmatrix} \Delta x,\ \Delta \dot{x},\ \Delta \theta,\ \Delta \dot{\theta} \end{bmatrix}^T \) is the state vector
    - \( Q \in \mathbb{R}^{4 \times 4} \) penalizes state deviations 
    - \( R > 0 \) penalizes the control effort


    The optimal control law is given by:

    \[
    \Delta \phi = -K_{oc} \cdot x
    \quad \text{with} \quad
    K_{oc} = R^{-1} B^T P
    \]

    where \( P \) is the unique, positive semi-definite solution of the algebraic Riccati equation:

    \[
    A^T P + P A - P B R^{-1} B^T P + Q = 0
    \]



    We aim to design an LQR controller such that:

    - \( \Delta x(t) \rightarrow 0 \) in about 20 seconds  
    - The system is asymptotically stable
    We Chosse for exemple:

    \[
    Q = \text{diag}(1, 0.1, 0.5, 0.05), \quad R = 1
    \]

    According to this choices , we  calculate K_{oc}, using the code below, and we get :
    \[
    K_{oc} = 
    \begin{bmatrix}
    1 \\
    2.41273157 \\
    -2.86063682 \\
    -2.12559926
    \end{bmatrix}
    \]



    We beleive that this Choice Ensures the Desired Behavior, because : 

    1. \( \Delta x(t) \rightarrow 0 \) in ~20 seconds and \( \Delta \theta(t) \rightarrow 0 \) in ~20 seconds

    - The high value \( q_x = 1 \) strongly penalizes deviations in horizontal position.
    - Increasing the value of \( q_{\dot{\theta}} \) in the matrix \( Q \) helps dampen oscillations in \( \Delta \theta(t) \).
    - \( q_{\dot{\theta}} \) = 0.5 penalizes the angular velocity \( \Delta \dot{\theta}(t) \).


    2. Asymptotic Stability of the System

    - The LQR formulation guarantees asymptotic stability, provided that the pair \( (A, B) \) is controllable.
    - With \( Q \succeq 0 \) (positive semi-definite) and \( R > 0 \), the Riccati equation admits a solution.
    - The resulting controller stabilizes the system.
    """
    )
    return


@app.cell
def _(J, g, inv, l, np, solve_continuous_are):
    # Matrices d'état du système linéarisé
    A_3 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_3 = np.array([
        [0],
        [-g],
        [0],
        [-l * g / J]
    ])

    # Pondérations pour LQR
    QQ = np.diag([1.0,   # Δx
                 0.1,   # Δx_dot
                 0.5,   # Δθ
                 0.05]) # Δθ_dot

    N = np.array([[1]])

    P= solve_continuous_are(A_3,B_3,QQ,N)
    K_oc = inv(N) @ B_3.T @ P
    K_oc

    return (K_oc,)


@app.cell
def _(J, K_oc, g, l, np, plt):
    from scipy.integrate import odeint
    # Fonction dynamique en boucle fermée
    def system_dynamics(X, f):
        dx, dx_dot, dtheta, dtheta_dot = X
        delta_phi = float(-K_oc @ X)  # Assure un scalaire
        dXdt = [
            dx_dot,
            -g * (dtheta + delta_phi),
            dtheta_dot,
            - (l * g / J) * delta_phi
        ]
        return dXdt

    # Conditions initiales
    X0 = [0, 0, np.pi / 4, 0]  # Déviation initiale d’angle

    # Intégration temporelle
    f = np.linspace(0, 30, 1000)
    solt = odeint(system_dynamics, X0, f)

    # Extraction des résultats
    dx = solt[:, 0]
    dx_dot = solt[:, 1]
    dtheta = solt[:, 2]
    dtheta_dot = solt[:, 3]
    dphi = (-K_oc @ solt.T).flatten()  # Convertir en vecteur 1D

    # Vérification de convergence de Δx(f)
    tolerance = 0.01  # Seuil de convergence
    t_95 = None
    for i in range(len(f)):
        if abs(dx[i]) < tolerance:
            t_95 = f[i]
            break

    if t_95 is not None:
        print(f"\nΔx(f) atteint {tolerance} à f = {t_95:.2f} secondes")
    else:
        print("\nΔx(f) ne converge pas suffisamment vite.")

    # Graphiques
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(f, dx, label=r'$\Delta x(f)$')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Temps (s)')
    plt.ylabel(r'$\Delta x(f)$')
    plt.title(r'$\Delta x(f)$ – Position horizontale')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(f, dtheta, 'r', label=r'$\Delta \theta(f)$')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Temps (s)')
    plt.ylabel(r'$\Delta \theta(f)$')
    plt.title(r'$\Delta \theta(f)$ – Orientation')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(f, dphi, 'g', label=r'$\Delta \phi(f)$')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Temps (s)')
    plt.ylabel(r'$\Delta \phi(f)$')
    plt.title(r'$\Delta \phi(f)$ – Commande appliquée')
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


app._unparsable_cell(
    r"""
    The following simulation demonstrates that the chosen parameters are valid for the non-linear model:
    """,
    name="_"
)


@app.cell
def _(J, M, g, l, np, solve_ivp):
    def redstartt_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)

            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (f * np.cos(theta + phi)) / M - g
            d2theta = (-l * np.sin(phi)) * f / J

            return [dx, d2x, dy, d2y, dtheta, d2theta]

        sol = solve_ivp(fun, t_span, y0, dense_output=True)
        return sol
    return


@app.cell
def _(J, K_oc, K_pp, g, l, np, plt, solve_ivp):
    def simulate_nonlinear(x0, T, dt, control_law):
        """
        Simulates the nonlinear booster system using a given control law.

        Args:
            x0: Initial state [Δx, Δẋ, Δθ, Δθ̇]
            T: Simulation duration (seconds)
            dt: Time step (seconds)
            control_law: function u(t, x) -> Δϕ(t)

        Returns:
            ts: Time array
            xs: State array
            us: Control inputs over time
        """
        def dynamics(t, x):
            dx, dx_dot, dtheta, dtheta_dot = x
            delta_phi = float(control_law(t, x))
            dx_ddot = -g * (dtheta + delta_phi)
            dtheta_ddot = - (l * g / J) * delta_phi
            return [dx_dot, dx_ddot, dtheta_dot, dtheta_ddot]

        ts = np.arange(0, T + dt, dt)
        sol = solve_ivp(dynamics, [0, T], x0, t_eval=ts)
        xs = sol.y.T
        us = np.array([control_law(t, x) for t, x in zip(ts, xs)])
        return ts, xs, us




    def u_pp(t, x):
        return -K_pp @ x

    def u_lqr(t, x):
        return -K_oc @ x
    xx = [1.0, 0.0, 0.5, 0.0]
    T = 30
    dt = 0.05

    # Simulate with both controllers
    ts_pp, xs_pp, us_pp = simulate_nonlinear(xx, T, dt, u_pp)
    ts_lqr, xs_lqr, us_lqr = simulate_nonlinear(xx, T, dt, u_lqr)

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(ts_pp, xs_pp[:, 0], label="Δx - Pole Placement")
    axes[0].plot(ts_lqr, xs_lqr[:, 0], label="Δx - LQR", linestyle='--')
    axes[0].set_ylabel("Δx(t)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(ts_pp, xs_pp[:, 2], label="Δθ - Pole Placement")
    axes[1].plot(ts_lqr, xs_lqr[:, 2], label="Δθ - LQR", linestyle='--')
    axes[1].set_ylabel("Δθ(t)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(ts_pp, us_pp, label="Δϕ - Pole Placement")
    axes[2].plot(ts_lqr, us_lqr, label="Δϕ - LQR", linestyle='--')
    axes[2].set_ylabel("Δϕ(t)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
