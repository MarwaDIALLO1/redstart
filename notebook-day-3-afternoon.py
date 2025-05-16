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

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, la, mpl, np, plt, sci, scipy, tqdm


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


@app.cell
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \begin{bmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y} \\
    \theta \\
    \dot{\theta} \\
    f \\
    \phi
    \end{bmatrix}
    =
    \begin{bmatrix}
    ? \\
    0 \\
    ? \\
    0 \\
    0 \\
    0 \\
    M g \\
    0
    \end{bmatrix}
    $$
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M (d/dt)^2 \Delta x &= - Mg (\Delta \theta + \Delta \phi)  \\
    M (d/dt)^2 \Delta y &= \Delta f \\
    J (d/dt)^2 \Delta \theta &= - (Mg \ell) \Delta \phi \\
    \end{align*}
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    A = 
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0  & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0  & 0 \\
    0 & 0 & 0 & 0 & 0  & 0 \\
    0 & 0 & 0 & 0 & 0  & 1 \\
    0 & 0 & 0 & 0 & 0  & 0 
    \end{bmatrix}
    \;\;\;
    B = 
    \begin{bmatrix}
    0 & 0\\ 
    0 & -g\\ 
    0 & 0\\ 
    1/M & 0\\
    0 & 0 \\
    0 & -M g \ell/J\\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(g, np):
    A = np.zeros((6, 6))
    A[0, 1] = 1.0
    A[1, 4] = -g
    A[2, 3] = 1.0
    A[4, -1] = 1.0
    A
    return (A,)


@app.cell(hide_code=True)
def _(J, M, g, l, np):
    B = np.zeros((6, 2))
    B[ 1, 1]  = -g 
    B[ 3, 0]  = 1/M
    B[-1, 1] = -M*g*l/J
    B
    return (B,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(A, la):
    # No since 0 is the only eigenvalue of A
    eigenvalues, eigenvectors = la.eig(A)
    eigenvalues
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


@app.cell(hide_code=True)
def _(A, B, np):
    # Controllability
    cs = np.column_stack
    mp = np.linalg.matrix_power
    KC = cs([mp(A, k) @ B for k in range(6)])
    KC
    return (KC,)


@app.cell(hide_code=True)
def _(KC, np):
    # Yes!
    np.linalg.matrix_rank(KC) == 6
    return


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
def _(J, M, g, l, np):
    A_lat = np.array([
        [0, 1, 0, 0], 
        [0, 0, -g, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0]], dtype=np.float64)
    B_lat = np.array([[0, -g, 0, - M * g * l / J]]).T

    A_lat, B_lat
    return A_lat, B_lat


@app.cell(hide_code=True)
def _(A_lat, B_lat, np):
    # Controllability
    _cs = np.column_stack
    _mp = np.linalg.matrix_power
    KC_lat = _cs([_mp(A_lat, k) @ B_lat for k in range(6)])
    KC_lat
    return (KC_lat,)


@app.cell(hide_code=True)
def _(KC_lat, np):
    np.linalg.matrix_rank(KC_lat) == 4
    return


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


@app.cell(hide_code=True)
def _(J, M, g, l, np):
    def make_fun_lat(phi):
        def fun_lat(t, state):
            x, dx, theta, dtheta = state
            phi_ = phi(t, state)
            #if linearized:
            d2x = -g * (theta + phi_)
            d2theta = -M * g * l / J * phi_
            #else:
            #d2x = -g * np.sin(theta + phi_)
            #d2theta = -M * g * l / J * np.sin(phi_)
            return np.array([dx, d2x, dtheta, d2theta])

        return fun_lat
    return (make_fun_lat,)


@app.cell(hide_code=True)
def _(make_fun_lat, mo, np, plt, sci):
    def lin_sim_1():
        def _phi(t, state):
            return 0.0
        _f_lat = make_fun_lat(_phi)
        _t_span = [0, 10]
        state_0 = [0, 0, 45 * np.pi/180.0, 0]
        _r = sci.solve_ivp(fun=_f_lat, y0=state_0, t_span=_t_span, dense_output=True)
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _sol_t = _r.sol(_t)
        _fig, (_ax1, _ax2) = plt.subplots(2, 1, sharex=True)
        _ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend()
        _ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.grid(True)
        _ax2.set_xlabel(r"time $t$")
        _ax2.legend()
        return mo.center(_fig)
    lin_sim_1()
    return


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


@app.cell(hide_code=True)
def _(A_lat, B_lat, make_fun_lat, mo, np, plt, sci):

    def lin_sim_2():
        # Manual tuning of K (Angle only)

        K = np.array([0.0, 0.0, -1.0, -1.0])

        print("eigenvalues:", np.linalg.eig(A_lat - B_lat.reshape((-1,1)) @ K.reshape((1, -1))).eigenvalues)

        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - K.dot(state)

        #_f_lat = make_fun_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) # , linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_2()
    return


@app.cell
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
def _(A_lat, B_lat, make_fun_lat, mo, np, plt, sci, scipy):
    Kpp = scipy.signal.place_poles(
        A=A_lat, 
        B=B_lat, 
        poles=1.0*np.array([-0.5, -0.51, -0.52, -0.53])
    ).gain_matrix.squeeze()

    def lin_sim_3():
        print(f"Kpp = {Kpp}")

        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - Kpp.dot(state)

        #_f_lat = make_f_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) # , linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_3()
    return (Kpp,)


@app.cell
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
def _(A_lat, B_lat, l, make_fun_lat, mo, np, plt, sci, scipy):
    _Q = np.zeros((4,4))
    _Q[0, 0] = 1.0
    _Q[1, 1] = 0.0
    _Q[2, 2] = (2*l)**2
    _Q[3, 3] = 0.0
    _R = 10*(2*l)**2 * np.eye(1)

    _Pi = scipy.linalg.solve_continuous_are(
        a=A_lat, 
        b=B_lat, 
        q=_Q, 
        r=_R
    )
    Koc = (np.linalg.inv(_R) @ B_lat.T @ _Pi).squeeze()
    print(f"Koc = {Koc}")

    def lin_sim_4():    
        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - Koc.dot(state)

        #_f_lat = make_fun_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) #, linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_4()
    return (Koc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    Kpp,
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
    def video_sim_Kpp():
        t_span = [0.0, 20.0]
        y0 = [0.0, 0.0, 20.0, 0.0, 45 * np.pi/180.0, 0.0]
        def f_phi(t, state):
            x, dx, y, dy, theta, dtheta = state  
            return np.array(
                [M*g, -Kpp.dot([x, dx, theta, dtheta])]
            )
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_Kpp.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +24*l)
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

    mo.video(src=video_sim_Kpp())

    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    Koc,
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
    def video_sim_Koc():
        t_span = [0.0, 20.0]
        y0 = [0.0, 0.0, 20.0, 0.0, 45 * np.pi/180.0, 0.0]
        def f_phi(t, state):
            x, dx, y, dy, theta, dtheta = state  
            return np.array(
                [M*g, -Koc.dot([x, dx, theta, dtheta])]
            )
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_Koc.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +24*l)
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

    mo.video(src=video_sim_Koc())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Exact Linearization


    Consider an auxiliary system which is meant to compute the force $(f_x, f_y)$ applied to the booster. 

    Its inputs are 

    $$
    v = (v_1, v_2) \in \mathbb{R}^2,
    $$

    its dynamics 

    $$
    \ddot{z} = v_1 \qquad \text{ where } \quad z\in \mathbb{R}
    $$ 

    and its output $(f_x, f_y) \in \mathbb{R}^2$ is given by

    \[
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} = R\left(\theta - \frac{\pi}{2}\right)
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    \]

    ⚠️ Note that the second component $f_y$ of the reactor force is undefined whenever $z=0$.

    Consider the output $h$ of the original system

    $$
    h := 
    \begin{bmatrix}
    x - (\ell/3) \sin \theta \\
    y + (\ell/3) \cos \theta
    \end{bmatrix} \in \mathbb{R}^2
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Geometrical Interpretation

    Provide a geometrical interpretation of $h$ (for example, make a drawing).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""🔓 The coordinates $h$ represent a fixed point on the booster. Start from the reactor, move to the center of mass (distance $\ell$) then continue for $\ell/3$ in this direction. The coordinates of this point are $h$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 First and Second-Order Derivatives

    Compute $\dot{h}$ as a function of $\dot{x}$, $\dot{y}$, $\theta$ and $\dot{\theta}$ (and constants) and then $\ddot{h}$ as a function of $\theta$ and $z$ (and constants) when the auxiliary system is plugged in the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 We have 

    $$
    \boxed{
    \dot{h} = 
    \begin{bmatrix}
    \dot{x} - (\ell /3)  (\cos \theta) \dot{\theta} \\
    \dot{y} - (\ell /3) (\sin \theta) \dot{\theta}
    \end{bmatrix}}
    $$

    and therefore

    \begin{align*}
    \ddot{h} &=
    \begin{bmatrix}
    \ddot{x} - (\ell/3)\cos\theta\, \ddot{\theta} + (\ell/3)\sin\theta\, \dot{\theta}^2 \\
    \ddot{y} - (\ell/3)\sin\theta\, \ddot{\theta} - (\ell/3)\cos\theta\, \dot{\theta}^2
    \end{bmatrix} \\
    &=
    \begin{bmatrix}
    \frac{f_x}{M} - \frac{\ell}{3} \cos\theta \cdot \frac{3}{M\ell} (\cos\theta\, f_x + \sin\theta\, f_y) + \frac{\ell}{3} \sin\theta\, \dot{\theta}^2 \\
    \frac{f_y}{M} - g - \frac{\ell}{3} \sin\theta \cdot \frac{3}{M\ell} (\cos\theta\, f_x + \sin\theta\, f_y) - \frac{\ell}{3} \cos\theta\, \dot{\theta}^2
    \end{bmatrix} \\
    &=
    \frac{1}{M}
    \begin{bmatrix}
    \sin\theta \\
    -\cos\theta
    \end{bmatrix}
    \left(
    \begin{bmatrix}
    \sin\theta & -\cos\theta
    \end{bmatrix}
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix}
    + M\frac{\ell}{3} \dot{\theta}^2
    \right)
    -
    \begin{bmatrix}
    0 \\
    g
    \end{bmatrix}
    \end{align*}


    On the other hand, since

    \[
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} = R\left(\theta - \frac{\pi}{2}\right)
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    \]

    we have

    $$
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    = R\left(\frac{\pi}{2} - \theta\right) \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} =
    \begin{bmatrix}
    \sin \theta & - \cos \theta \\
    \cos \theta & \sin \theta
    \end{bmatrix}
    \begin{bmatrix}
    f_x \\ f_y
    \end{bmatrix}
    $$

    and therefore we end up with

    $$
    \boxed{\ddot{h} = 
      \frac{1}{M}
      \begin{bmatrix}
        \sin\theta \\
        -\cos\theta
       \end{bmatrix}
      z
      -
      \begin{bmatrix}
        0 \\
        g
      \end{bmatrix}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Third and Fourth-Order Derivatives 

    Compute the third derivative $h^{(3)}$ of $h$ as a function of $\theta$ and $z$ (and constants) and then the fourth derivative $h^{(4)}$ of $h$ with respect to time as a function of $\theta$, $\dot{\theta}$, $z$, $\dot{z}$, $v$ (and constants) when the auxiliary system is on.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 We have 

    \[
    \boxed{
    h^{(3)} = \frac{1}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}z + \frac{1}{M}
    \begin{bmatrix}
    \sin \theta \\
    -\cos \theta
    \end{bmatrix}
    \dot{z}
    }
    \]

    and consequently

    \[
    \begin{aligned}
    h^{(4)} &= \frac{1}{M}
    \begin{bmatrix}
    -\sin \theta \\
    \cos \theta
    \end{bmatrix}
    \dot{\theta}^2 z + \frac{1}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \frac{3}{Ml} (\cos \theta f_x + \sin \theta f_y) z \\
    &+ \frac{2}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}\dot{z} + \frac{1}{M}
    \begin{bmatrix}
    \sin \theta \\
    -\cos \theta
    \end{bmatrix}
    v_1
    \end{aligned}
    \]

    Since

    \[
    \begin{bmatrix}
    z - \frac{Ml}{3} \dot{\theta}^2 \\
    \frac{Mlv_2}{3z}
    \end{bmatrix}
    = R\left(\frac{\pi}{2} - \theta\right) \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} =
    \begin{bmatrix}
    \sin \theta f_x - \cos \theta f_y \\
    \cos \theta f_x + \sin \theta f_y
    \end{bmatrix}
    \]

    we have

    \[
    h^{(4)} = \frac{1}{M}
    \begin{bmatrix}
    \sin \theta & \cos \theta \\
    -\cos \theta & \sin \theta
    \end{bmatrix}
    \begin{bmatrix}
    v_1 \\
    v_2
    \end{bmatrix}
    + \frac{1}{M}
    \begin{bmatrix}
    -\sin \theta \\
    \cos \theta
    \end{bmatrix}
    \dot{\theta}^2 z
    + \frac{2}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}\dot{z}
    \]

    \[
    \boxed{
    h^{(4)}
    = \frac{1}{M} R \left( \theta - \frac{\pi}{2} \right)
    \left(
    v +
    \begin{bmatrix}
    -\dot{\theta}^2 z \\
    2 \dot{\theta} \dot{z}
    \end{bmatrix}
    \right)
    }
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Exact Linearization

    Show that with yet another auxiliary system with input $u=(u_1, u_2)$ and output $v$ fed into the previous one, we can achieve the dynamics

    $$
    h^{(4)} = u
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 Since

    \[
    h^{(4)}
    = \frac{1}{M} R \left( \theta - \frac{\pi}{2} \right)
    \left(
    v +
    \begin{bmatrix}
    -\dot{\theta}^2 z \\
    2 \dot{\theta} \dot{z}
    \end{bmatrix}
    \right)
    \]  

    we can define $v$ as 

    $$
    \boxed{
    v =
    M \, R \left(\frac{\pi}{2} - \theta \right)
    u + 
    \begin{bmatrix}
    \dot{\theta}^2 z \\
    -2 \dot{\theta} \dot{z}
    \end{bmatrix}
    }
    $$

    and achieve the desired result.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 State to Derivatives of the Output

    Implement a function `T` of `x, dx, y, dy, theta, dtheta, z, dz` that returns `h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y`.
    """
    )
    return


@app.cell
def _(M, g, l, np):
    def T(x, dx, y, dy, theta, dtheta, z, dz):

        # Position h
        h_x = x - (l/3) * np.sin(theta)
        h_y = y + (l/3) * np.cos(theta)
    
        # First derivative of h
        dh_x = dx - (l/3) * np.cos(theta) * dtheta
        dh_y = dy - (l/3) * np.sin(theta) * dtheta
    
        # Second derivative of h
        d2h_x = (1/M) * np.sin(theta) * z 
        d2h_y = (1/M) * (-np.cos(theta)) * z - g
    
        # Third derivative of h
        d3h_x = (1/M) * (np.cos(theta) * dtheta * z + np.sin(theta) * dz)
        d3h_y = (1/M) * (np.sin(theta) * dtheta * z - np.cos(theta) * dz)
    
        return h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y
    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Inversion 


    Assume for the sake of simplicity that $z<0$ at all times. Show that given the values of $h$, $\dot{h}$, $\ddot{h}$ and $h^{(3)}$, one can uniquely compute the booster state (the values of $x$, $\dot{x}$, $y$, $\dot{y}$, $\theta$, $\dot{\theta}$) and auxiliary system state (the values of $z$ and $\dot{z}$).

    Implement the corresponding function `T_inv`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1. Position \( h \)
    The position \( h \) is given by:

    $$
    h = \begin{bmatrix} x - \frac{\ell}{3} \sin \theta \\ y + \frac{\ell}{3} \cos \theta \end{bmatrix}
    $$

    From this, we can express \( x \) and \( y \) as:

    $$
    x = h_x + \frac{\ell}{3} \sin \theta, \quad y = h_y - \frac{\ell}{3} \cos \theta
    $$

    ### 2. First Derivative \( \dot{h} \)
    The first derivative \( \dot{h} \) is:

    $$
    \dot{h} = \begin{bmatrix} \dot{x} - \frac{\ell}{3} \cos \theta \dot{\theta} \\ \dot{y} - \frac{\ell}{3} \sin \theta \dot{\theta} \end{bmatrix}
    $$

    From this, we can express \( \dot{x} \) and \( \dot{y} \) as:

    $$
    \dot{x} = \dot{h}_x + \frac{\ell}{3} \cos \theta \dot{\theta}, \quad \dot{y} = \dot{h}_y + \frac{\ell}{3} \sin \theta \dot{\theta}
    $$

    ### 3. Second Derivative \( \ddot{h} \)
    The second derivative \( \ddot{h} \) is:

    $$
    \ddot{h} = \frac{1}{M} \begin{bmatrix} \sin \theta \\ -\cos \theta \end{bmatrix} z - \begin{bmatrix} 0 \\ g \end{bmatrix}
    $$

    From the \( x \)-component:

    $$
    \ddot{h}_x = \frac{1}{M} \sin \theta z \implies z = \frac{M \ddot{h}_x}{\sin \theta}
    $$

    Since \( z < 0 \) and \( \sin\theta \) can be positive or negative,  \( \theta \) must be in a quadrant where \( \sin\theta \ne 0 \).

    From the \( y \)-component:

    $$
    \ddot{h}_y = -\frac{1}{M} \cos \theta z - g \implies \cot \theta = -\frac{\ddot{h}_y + g}{\ddot{h}_x}
    $$

    Thus:

    $$
    \theta = \text{arccot}\left( -\frac{\ddot{h}_y + g}{\ddot{h}_x} \right)
    $$

    ### 4. Third Derivative \( h^{(3)} \)
    The third derivative \( h^{(3)} \) is:

    $$
    h^{(3)} = \frac{1}{M} \begin{bmatrix} \cos \theta \\ \sin \theta \end{bmatrix} \dot{\theta} z + \frac{1}{M} \begin{bmatrix} \sin \theta \\ -\cos \theta \end{bmatrix} \dot{z}
    $$

    Solving for \( \dot{z} \) and \( \dot{\theta} \):

    $$
    \dot{z} = M (\sin \theta h^{(3)}_x - \cos \theta h^{(3)}_y)
    $$

    $$
    \dot{\theta} = \frac{M (h^{(3)}_x \cos \theta + \sin \theta h^{(3)}_y)}{z}
    $$
    """
    )
    return


@app.cell
def _(M, g, l, np):
    def T_inv(h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y):
    
        # Solve for theta
        theta = np.arctan2(-d2h_x, d2h_y + g) 
    
        # Solve for z
        sin_theta = np.sin(theta)
        z = (M * d2h_x) / sin_theta
    
        # Solve for dot{z}
        dz = M * (sin_theta * d3h_x - np.cos(theta) * d3h_y)
    
        # Solve for dot{theta}
        dtheta = (M * (d3h_x * np.cos(theta) + sin_theta * d3h_y)) / z
    
        # Solve for x and y
        x = h_x + (l/3) * sin_theta
        y = h_y - (l/3) * np.cos(theta)
    
        # Solve for dx and dy
        dx = dh_x + (l/3) * np.cos(theta) * dtheta
        dy = dh_y + (l/3) * sin_theta * dtheta
    
        return x, dx, y, dy, theta, dtheta, z, dz
    return (T_inv,)


@app.cell
def _(M, T, T_inv, g, l, np):
    h_x01, h_y01 = 0.5, 1.0
    dh_x01, dh_y01 = 0.2, -0.1
    d2h_x01, d2h_y01 = 3.0, -5.0
    d3h_x01, d3h_y01 = -1.0, 0.5

    # On applique T_inv
    x1, dx1, y1, dy1, theta1, dtheta1, z1, dz1 = T_inv(h_x01, h_y01, dh_x01, dh_y01, d2h_x01, d2h_y01, d3h_x01, d3h_y01, M, l, g)

    # Puis on applique T
    h_x1, h_y1, dh_x1, dh_y1, d2h_x1, d2h_y1, d3h_x1, d3h_y1 = T(x1, dx1, y1, dy1, theta1, dtheta1, z1, dz1)

    # Affichage des résultats
    print("Original : ", np.round([h_x01, h_y01, dh_x01, dh_y01, d2h_x01, d2h_y01, d3h_x01, d3h_y01], 5))
    print("After T ∘ T_inv : ", np.round([h_x1, h_y1, dh_x1, dh_y1, d2h_x1, d2h_y1, d3h_x1, d3h_y1], 5))

    # Erreur absolue
    error = np.abs(np.array([h_x1, h_y1, dh_x1, dh_y1, d2h_x1, d2h_y1, d3h_x1, d3h_y1]) - 
                   np.array([h_x01, h_y01, dh_x01, dh_y01, d2h_x01, d2h_y01, d3h_x01, d3h_y01]))

    print("Erreur absolue : ", np.round(error, 5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Admissible Path Computation

    Implement a function

    ```python
    def compute(
        x_0,
        dx_0,
        y_0,
        dy_0,
        theta_0,
        dtheta_0,
        z_0,
        dz_0,
        x_tf,
        dx_tf,
        y_tf,
        dy_tf,
        theta_tf,
        dtheta_tf,
        z_tf,
        dz_tf,
        tf,
    ):
        ...

    ```

    that returns a function `fun` such that `fun(t)` is a value of `x, dx, y, dy, theta, dtheta, z, dz, f, phi` at time `t` that match the initial and final values provided as arguments to `compute`.
    """
    )
    return


@app.cell
def _(M, T, l, np):
    from scipy.interpolate import CubicSpline

    def compute(x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0,
                x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf, tf):
    
        # Interpolation des états avec splines cubiques
        splines = {
            'x': CubicSpline([0, tf], [x_0, x_tf], bc_type=((1, dx_0), (1, dx_tf))),
            'y': CubicSpline([0, tf], [y_0, y_tf], bc_type=((1, dy_0), (1, dy_tf))),
            'theta': CubicSpline([0, tf], [theta_0, theta_tf], bc_type=((1, dtheta_0), (1, dtheta_tf))),
            'z': CubicSpline([0, tf], [z_0, z_tf], bc_type=((1, dz_0), (1, dz_tf)))
        }

        def fun(t):
            # États interpolés
            x = splines['x'](t)
            dx = splines['x'].derivative(1)(t)
            y = splines['y'](t)
            dy = splines['y'].derivative(1)(t)
            theta = splines['theta'](t)
            dtheta = splines['theta'].derivative(1)(t)
            z = splines['z'](t)
            dz = splines['z'].derivative(1)(t)
        
            # Calcul des termes dynamiques (remplacez T par votre vraie fonction)
            # Hypothèse: T retourne un array où les éléments 6 et 7 sont dh3_x et dh3_y
            H = T(x, dx, y, dy, theta, dtheta, z, dz)
            dh3_x = H[6]
            dh3_y = H[7]
        
            # Dérivées temporelles
            dh4_x = (T(x + 1e-6, dx, y, dy, theta, dtheta, z, dz)[6] - dh3_x) / 1e-6  # Approximation numérique
            dh4_y = (T(x, dx, y + 1e-6, dy, theta, dtheta, z, dz)[7] - dh3_y) / 1e-6
        
            u = np.array([dh4_x, dh4_y])
        
            # Matrice de rotation R(π/2 - θ)
            angle_1 = np.pi/2 - theta
            R = np.array([
                [np.cos(angle_1), -np.sin(angle_1)],
                [np.sin(angle_1),  np.cos(angle_1)]
            ])
        
            # Terme dynamique
            dynamic_term = np.array([
                dtheta**2 * z,
                -2 * dtheta * dz
            ])
        
            # Calcul final de v
            v = M * R @ u + dynamic_term
        
            # Calcul de f_x et f_y
            angle_2 = theta - np.pi/2
            rotation_matrix_2 = np.array([
                [np.cos(angle_2), -np.sin(angle_2)],
                [np.sin(angle_2),  np.cos(angle_2)]
            ])
        
            term_1 = z - (M * l*2 * dtheta*2) / 3
            term_2 = z - (M * l * v[1]**2) / (3 * z)
            fx_fy = rotation_matrix_2 @ np.array([term_1, term_2])  
        
            f = np.sqrt(fx_fy[0]*2 + fx_fy[1]*2)  # Norme de la force
            phi = np.arctan2(fx_fy[1], fx_fy[0])     # Angle de la force

            return [x, dx, y, dy, theta, dtheta, z, dz, f, phi]
 
        return fun
    return (compute,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Graphical Validation

    Test your `compute` function with

      - `x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0 = 5.0, 0.0, 20.0, -1.0, -np.pi/8, 0.0, -M*g, 0.0`,
      - `x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf = 0.0, 0.0, 4/3*l, 0.0,     0.0, 0.0, -M*g, 0.0`,
      - `tf = 10.0`.

    Make the graph of the relevant variables as a function of time, then make a video out of the same result. Comment and iterate if necessary!
    """
    )
    return


@app.cell
def _(M, g, l, np):
    x_0, dx_0 = 5.0, 0.0
    y_0, dy_0 = 20.0, -1.0
    theta_0, dtheta_0 = -np.pi/8, 0.0
    z_0, dz_0 = -M*g, 0.0

    # État final
    x_tf, dx_tf = 0.0, 0.0
    y_tf, dy_tf = 4/3 * l, 0.0
    theta_tf, dtheta_tf = 0.0, 0.0
    z_tf, dz_tf = -M*g, 0.0

    # Temps final
    tf = 10.0
    return (
        dtheta_0,
        dtheta_tf,
        dx_0,
        dx_tf,
        dy_0,
        dy_tf,
        dz_0,
        dz_tf,
        tf,
        theta_0,
        theta_tf,
        x_0,
        x_tf,
        y_0,
        y_tf,
        z_0,
        z_tf,
    )


@app.cell
def _(np, plt):
    def plot_trajectory(fun, tf=10.0):
        t = np.linspace(0, tf, 500)
    
        # Récupération des valeurs à chaque instant
        x_list, dx_list = [], []
        y_list, dy_list = [], []
        theta_list, dtheta_list = [], []
        z_list, dz_list = [], []
        f_list, phi_list = [], []

        for time in t:
            x, dx, y, dy, theta, dtheta, z, dz, f, phi = fun(time)
            x_list.append(x)
            dx_list.append(dx)
            y_list.append(y)
            dy_list.append(dy)
            theta_list.append(theta)
            dtheta_list.append(dtheta)
            z_list.append(z)
            dz_list.append(dz)
            f_list.append(f)
            phi_list.append(phi)

        # Tracé
        fig, axs = plt.subplots(3, 3, figsize=(16, 10))
        fig.suptitle("Évolution temporelle de la trajectoire", fontsize=16)

        axs[0, 0].plot(t, x_list, label="x(t)")
        axs[0, 0].set_title("Position x(t)")
        axs[0, 0].grid(True)

        axs[0, 1].plot(t, y_list, label="y(t)", color='orange')
        axs[0, 1].set_title("Position y(t)")
        axs[0, 1].grid(True)

        axs[0, 2].plot(t, theta_list, label="θ(t)", color='green')
        axs[0, 2].set_title("Angle θ(t)")
        axs[0, 2].grid(True)

        axs[1, 0].plot(t, dx_list, label="dx(t)")
        axs[1, 0].set_title("Vitesse dx(t)")
        axs[1, 0].grid(True)

        axs[1, 1].plot(t, dy_list, label="dy(t)", color='orange')
        axs[1, 1].set_title("Vitesse dy(t)")
        axs[1, 1].grid(True)

        axs[1, 2].plot(t, dtheta_list, label="dθ(t)", color='green')
        axs[1, 2].set_title("Vitesse angulaire dθ(t)")
        axs[1, 2].grid(True)

        axs[2, 0].plot(t, z_list, label="z(t)", color='purple')
        axs[2, 0].set_title("Variable z(t)")
        axs[2, 0].grid(True)

        axs[2, 1].plot(t, f_list, label="f(t)", color='red')
        axs[2, 1].set_title("Force f(t)")
        axs[2, 1].grid(True)

        axs[2, 2].plot(t, phi_list, label="φ(t)", color='brown')
        axs[2, 2].set_title("Angle φ(t)")
        axs[2, 2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    return (plot_trajectory,)


@app.cell
def _(FFMpegWriter, FuncAnimation, draw_booster, l, np, plt, tqdm):
    def make_videoo(fun, tf=10.0, filename="booster_landing.mp4"):
        fig, ax = plt.subplots(figsize=(10, 8))
        num_frames = 300
        fps = 30
        t_vals = np.linspace(0, tf, num_frames)

        def update(frame):
            ax.clear()
            t = t_vals[frame]
            x, dx, y, dy, theta, dtheta, z, dz, f, phi = fun(t)
            draw_booster(x=x, y=y, theta=theta, f=f, phi=phi, axes=ax)
            ax.set_xlim(-4*l, 4*l)
            ax.set_ylim(-2*l, 24*l)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.set_title(f"Temps: {t:.2f}s")

        pbar = tqdm(total=num_frames, desc="Génération de la vidéo")
        ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
        writer = FFMpegWriter(fps=fps)
        ani.save(filename, writer=writer)
        pbar.close()
        print(f"Vidéo sauvegardée sous {filename}")
    return (make_videoo,)


@app.cell
def _(
    compute,
    dtheta_0,
    dtheta_tf,
    dx_0,
    dx_tf,
    dy_0,
    dy_tf,
    dz_0,
    dz_tf,
    make_videoo,
    plot_trajectory,
    tf,
    theta_0,
    theta_tf,
    x_0,
    x_tf,
    y_0,
    y_tf,
    z_0,
    z_tf,
):
    fun = compute(
        x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0,
        x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf,
        tf
    )

    # Affichage des graphiques
    plot_trajectory(fun, tf)

    # Génération de la vidéo
    make_videoo(fun, tf=tf)
    return


@app.cell
def _(booster_landing, images, mo, public):
    mo.video(src=public/images/booster_landing.mp4)
    return


if __name__ == "__main__":
    app.run()
