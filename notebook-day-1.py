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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci
    from scipy.integrate import solve_ivp
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, np, plt, solve_ivp, tqdm


@app.cell
def _():
    from matplotlib.patches import Rectangle, Polygon
    import matplotlib.transforms as transforms
    return Polygon, Rectangle, transforms


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


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


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


@app.cell
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
    (mo.video(src=_filename))
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


@app.cell
def _():
    g = 1
    M = 1
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


@app.cell
def _(mo):
    mo.center(mo.image(src="public/images/image (13).png"))
    return


@app.cell
def _(mo):
    mo.center(mo.image(src="public/images/image (14).png"))
    return


@app.cell
def _(mo):
    mo.center(mo.image(src="public/images/image (15).png"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    When expressing the force applied by the reactor in the reference of the booster, we obtain :

    $$
    \vec{F}_{\text{local}} = \begin{bmatrix} -f \sin \phi \\ f \cos \phi \end{bmatrix}.
    $$  

    To express it in the global reference (the fixed reference related to earth), we
    multiply the latter vector by the rotation matrix : 

    $$
     \vec{F}_{\text{global}} = R(\theta) \cdot \vec{F}_{\text{local}} = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \begin{bmatrix} -f \sin \phi \\ f \cos \phi \end{bmatrix}
    $$

    Thus,

    $$
     f_x = -f \cos \theta \cos \phi - f \sin \theta \sin \phi = -f \sin(\theta + \phi)
    $$

    $$
    f_y = -f \sin \theta \sin \phi + f \cos \theta \cos \phi = f \cos (\theta + \phi)
    $$
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


@app.cell
def _(mo):
    mo.md(
        r"""
    By applying Newton's second law to the booster, we get:

    $$
    M \cdot \frac{d\vec{v}}{dt} = \vec{F} + \vec{P}
    $$

    where:  
    - \( \vec{F}\) is the force generated by the reactor.  
    - \( \vec{P} \) is the weight of the booster.  

    Projecting this equation onto the global \( (\vec{x_0}, \vec{y_0}) \)coordinate system, we get:

    $$
    \begin{cases}
    M \cdot \ddot{x}(t) = f_x \\
    M \cdot \ddot{y}(t) = f_y - M \cdot g
    \end{cases}
    $$

    where:  
    \( \ddot{x}(t), \ddot{y}(t) \) are the accelerations of the center of mass.  



    And according to the previous question, we have :

    $$
    \begin{cases}
    f_x = - f \cdot \sin (\theta + \varphi) \\
    f_y = f \cdot \cos(\theta + \varphi)
    \end{cases}
    $$


    We obtain the ordinary differential equations (ODEs) that govern the position \( (x(t), y(t)) \) of the center of mass.:

    $$
    \begin{cases}
    \ddot{x}(t) = - \frac{f}{M} \cdot \sin (\theta + \varphi) \\
    \ddot{y}(t) = \frac{f}{M} \cdot \cos(\theta + \varphi) - g
    \end{cases}
    $$
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
def _(mo):
    mo.md(
        r"""
    \[
    J = J_{\text{center}} + m d^2
    \]

    ---

    ###  Application to the booster :

    - The moment of inertia around the center is:

    \[
    J_{\text{center}} = \frac{1}{12} M (2l)^2 = \frac{1}{3} M l^2
    \]
    """
    )
    return


@app.cell
def _(M, l):
    J = (4/3) * M * l**2
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


@app.cell
def _(mo):
    mo.md(
        r"""
    According to the moments theorem :

    $$\sum \tau = I\ddot{\theta}$$

    where:

    $\sum \tau$ is the sum of all torques about the center of mass
    $I$ is the moment of inertia
    $\ddot{\theta}$ is the angular acceleration


    ### Calculation of Torques

    1. **Gravitational Force Torque**: 
       Since gravity acts at the center of mass, its moment arm for rotation about the center is zero. However, if we consider the effect of gravity when the booster is tilted, we need to analyze the effective moment.

       When the booster is tilted at angle $\theta$, the center of mass is displaced from the vertical axis. The gravitational force creates a torque that tries to restore the vertical position.

       The moment arm for this torque is $\ell\sin\theta$, and the torque is:
       $$\tau_g = -Mg\ell\sin\theta$$

       The negative sign indicates that this torque opposes an increase in $\theta$.

    2. **Reactor Force Torque**:


       $$\vec{M}_{I}= \vec{M}_{centermass}+\vec{IA}\wedge \vec{F}
                    =\vec{0}-l \vec{y}_1\wedge (-fsin \phi \vec{y}_1 +fcos \phi  \vec{x}_1)$$

       $$\implies \tau_r =f\ell\sin\phi$$

    ### Total Torque and Differential Equation

    The total torque is the sum of all torques:
    $$\sum \tau = \tau_g + \tau_r = -Mg\ell\sin\theta + f\ell\sin\phi$$

    Applying the moments theorem:
    $$\sum \tau = I\ddot{\theta}$$

    Therefore:
    $$I\ddot{\theta} = -Mg\ell\sin\theta + f\ell\sin\phi$$


    The final differential equation is:

    $$\ddot{\theta} = -\frac{3g}{2\ell}\sin\theta + \frac{3f}{2M\ell}\sin\phi$$
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


@app.cell
def _():
    return


@app.cell
def _(J, M, g, l, np, solve_ivp):
    def redstart_solve(t_span, y0, f_phi):
        def dynamics(t, y):
            # État 
            x, dx, y_pos, dy, theta, dtheta = y

            f, phi = f_phi(t, y)

            fx = -f * np.sin(theta + phi)
            fy = f * np.cos(theta + phi) - M * g  

            # Accélérations
            ddx = fx / M
            ddy = fy / M

            # Moment autour du point I
            torque = l * f * np.sin(phi)
            ddtheta = torque / J

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        # solve_ivp
        sol_ivp = solve_ivp(dynamics, t_span, y0, dense_output=True)


        def sol(t):
            return sol_ivp.sol(t)

        return sol


    return (redstart_solve,)


@app.cell
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

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


app._unparsable_cell(
    r"""
    We are given the vertical position (in meters) of a booster with mass \( m = 1 \, \mathrm{kg} \), governed by the following differential equation according to Newton's second law:

    $$
    \ddot{y}(t) = \frac{f(t)}{M} - g,
    $$

    At the initial time \( t = 0 \), the initial conditions are:  

    - \( y(0) = 10 \),  
    - \( \dot{y}(0) = v_0 \),  
    - \( \dot{x}(0) = 0 \),  
    - \( \theta(0) = 0 \),  
    - \( \dot{\theta}(0) = 0 \).

    We want to find a time-varying force \( f(t) \) such that:

    - \( y(5) = \ell \) (landing at final altitude),  
    - \( \dot{y}(5) = 0 \) (zero velocity at landing).



    First, we need to find a reference trajectory \( y(t) \) satisfying the boundary conditions:

    $$
    \begin{cases}
    y(0) = 10, \\
    \dot{y}(0) = v_0, \\
    y(5) =\ell \, \\
    \dot{y}(5) = 0.
    \end{cases}
    $$

    Assume a cubic polynomial form for \( y(t) \):

    $$
    y(t) = a t^3 + b t^2 + c t + d.
    $$

    Imposing the four conditions, we get the following system of equations:

    $$
    \begin{cases}
    d = 10, \\
    c = v_0, \\
    125a + 25b + 5c + d = \ell \ \\
    75a + 10b + c = 0.
    \end{cases}
    $$

    Solving this system, we find:  
    - \( a = \frac{1}{25} v_0 + 10 - \ell \),  
    - \( b = -\frac{100}{3} - \frac{2}{5} v_0 + \frac{17}{5} \ell \),  
    - \( c = v_0 \).


    From the second derivative of \( y(t) \), we get:

    $$
    \ddot{y}(t) = 6 a t + 2b.
    $$


    We can now compute the required force \( f(t) \) as:

    $$
    f(t) = \left( \frac{6}{25} v_0 + 60 - 6\ell \right) t + \left( -\frac{200}{3} - \frac{4}{5} v_0 + \frac{34}{5} \ell \right) + 9.81.
    $$

    This force ensures that the booster follows the desired trajectory exactly.
    """,
    name="_"
)


@app.cell
def _(M, g, np, solve_ivp):

    ell = 1  # half-length of booster (m)
    y0 = 10  # initial height (m)
    v0 = 0  # initial velocity (m/s)
    t_final = 5  # final time (s)

    # Define the cubic trajectory coefficients
    def get_trajectory_coefficients(v0, y0, y_final, t_final):
        """
        Calculate the coefficients of a cubic polynomial y(t) = at³ + bt² + ct + d
        that satisfies the boundary conditions:
        y(0) = y0, y'(0) = v0, y(t_final) = y_final, y'(t_final) = 0
        """
        d = y0
        c = v0

        # These formulas come from solving the system of linear equations:
        # d = y0
        # c = v0
        # a*t_final³ + b*t_final² + c*t_final + d = y_final
        # 3a*t_final² + 2b*t_final + c = 0
        a = (2*(y0 - y_final) + t_final*(v0 + 0)) / (t_final**3)
        b = (3*(y_final - y0) - t_final*(2*v0 + 0)) / (t_final**2)

        return a, b, c, d

    # Get trajectory coefficients
    a, b, c, d = get_trajectory_coefficients(v0, y0, ell, t_final)

    # Define the trajectory and its derivatives
    def y_trajectory(t):
        return a * t*3 + b * t*2 + c * t + d

    def dy_trajectory(t):
        return 3 * a * t**2 + 2 * b * t + c

    def ddy_trajectory(t):
        return 6 * a * t + 2 * b

    # Calculate the required force
    def f(t):
        return M * (ddy_trajectory(t) + g)

    # ODE for the system
    def system(t, state):
        # state = [y, dy/dt]
        y, dy = state

        # Calculate force at current time
        force = f(t)

        # Equations of motion
        dy_dt = dy
        ddy_dt = force/M - g

        return [dy_dt, ddy_dt]

    # Initial state [y(0), y'(0)]
    initial_state = [y0, v0]

    # Time points
    t_span = (0, t_final)
    t_eval = np.linspace(0, t_final, 100)

    # Solve the ODE
    solution = solve_ivp(system, t_span, initial_state, t_eval=t_eval, method='RK45')

    # Extract the solution
    t = solution.t
    y = solution.y[0]
    dy = solution.y[1]



    # Print final state and force equation
    print(f"Final position: {y[-1]:.6f} m (Target: {ell} m)")
    print(f"Final velocity: {dy[-1]:.6f} m/s (Target: 0 m/s)")
    print(f"Force equation: f(t) = {M*6*a:.6f}*t + {M*2*b:.6f} + {M*g:.6f}")
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


@app.cell
def _(Polygon, Rectangle, plt, transforms):


    def draw_rocket(x, y, theta, thrust_force, phi, M=1, g=1, l=1, ax=None):

        # Import inside function to avoid global redefinition 
        # in Marimo environment


        # Create axes if not provided
        if ax is None:
            ax = plt.gca()

        # Booster dimensions - length is now 2*l
        booster_length = 2 * l
        booster_width = l/2  # Ajusté pour être proportionnel à l

        # Draw landing zone at (0, -20)
        landing_zone = Rectangle((-5, -25), 10, 1, color='orange', alpha=0.7)
        ax.add_patch(landing_zone)

        # Calculate flame length (proportional to thrust)
        flame_length = (thrust_force / (M * g)) * l if (M * g) != 0 else 0
        flame_width = booster_width * 0.8

        # Create a transformation for the booster orientation

        t = transforms.Affine2D().rotate(theta).translate(x, y)

        # Draw the booster body (black rectangle)
        booster_x = -booster_width / 2
        booster_y = -booster_length / 2
        booster = Rectangle((booster_x, booster_y), booster_width, booster_length, 
                            color='black', transform=t + ax.transData)
        ax.add_patch(booster)

        # Calculate flame orientation (relative to booster orientation)
        flame_angle = phi  # phi is relative to the booster's orientation

        # Create flame shape as a triangle/polygon
        flame_t = transforms.Affine2D().rotate(flame_angle).translate(0, -booster_length/2).rotate(theta).translate(x, y)
        # Use numpy locally to avoid global import
        import numpy
        flame_points = numpy.array([
            [-flame_width/2, 0],
            [flame_width/2, 0],
            [0, -flame_length]
        ])
        flame = Polygon(flame_points, closed=True, color='red', alpha=0.8, 
                        transform=flame_t + ax.transData)
        ax.add_patch(flame)

        # Set appropriate limits and aspect ratio
        ax.set_xlim(-20, 20)
        ax.set_ylim(-25, 20)  # Ajusté pour mieux voir la zone d'atterrissage
        ax.grid(True)
        ax.set_aspect('equal')

        # Set labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        return ax

    # Example usage:
    if __name__ == "__main__":

        plt.figure(figsize=(6, 10))
        ax = plt.gca()

        # Example parameters with renamed variables to avoid conflicts
        pos_x = 0.5      # x position
        pos_y = 0        # y position
        theta = 0.2      # orientation angle in radians
        thrust = 5       # thrust force with renamed variable
        phi = 0.2        # thrust direction (relative to booster orientation) in radians

        draw_rocket(pos_x, pos_y, theta, thrust, phi)
        plt.title("Rocket Booster Visualization")
        plt.tight_layout()
        plt.show()
    return (draw_rocket,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_rocket,
    g,
    l,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def create_booster_videos():

        # Common parameters
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]

        # Scenario 1: Free fall (f=0, phi=0)
        def f_phi_1(t, y):
            return np.array([0.0, 0.0])

        # Scenario 2: Hovering (f=Mg, phi=0)
        def f_phi_2(t, y):
            return np.array([M*g, 0.0])

        # Scenario 3: Tilted thrust (f=Mg, phi=pi/8)
        def f_phi_3(t, y):
            return np.array([M*g, np.pi/8])

        # Scenario 4: Controlled landing
        def f_phi_4(t, y):
            # Calculate coefficients for cubic trajectory
            y_final = l
            a = (2*(y0[2] - y_final) + t_span[1]*(y0[3])) / (t_span[1]**3)
            b = (3*(y_final - y0[2]) - t_span[1]*(2*y0[3])) / (t_span[1]**2)
            c = y0[3]

            # Calculate required acceleration and force
            ddy = 6*a*t + 2*b
            f = M * (ddy + g)

            return np.array([f, 0.0])

        # Function to create a single video
        def create_video(filename, f_phi_func, title):
            # Solve trajectory
            sol = redstart_solve(t_span, y0, f_phi_func)

            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()

            # Simulation time points
            num_frames = 100
            t_frames = np.linspace(t_span[0], t_span[1], num_frames)

            # Calculate trajectory bounds for setting plot limits
            t_eval = np.linspace(t_span[0], t_span[1], 200)
            states = sol(t_eval)
            x_min, x_max = min(states[0]) - 2*l, max(states[0]) + 2*l
            y_min, y_max = min(min(states[2]) - 2*l, -1), max(states[2]) + 2*l

            # Ensure reasonable bounds
            x_min, x_max = min(x_min, -5), max(x_max, 5)
            y_min, y_max = min(y_min, -5), max(y_max, 15)

            pbar = tqdm(total=num_frames, desc=f"Generating {title}")

            def animate(frame_idx):
                ax.clear()

                # Set limits and title
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_title(f"{title} - t={t_frames[frame_idx]:.2f}s")
                ax.grid(True)

                # Get state for current frame
                t = t_frames[frame_idx]
                state = sol(t)
                x, dx, y_pos, dy, theta, dtheta = state

                # Get force and phi for current frame
                f, phi = f_phi_func(t, state)

                # Draw booster
                draw_rocket(x, y_pos, theta, f, phi, ax=ax)

                # Plot ground line
                ax.axhline(y_min, color='green', linestyle='-', alpha=0.3)
                ax.text(x_min + 1, y_min+0.5, "Ground", color='green')

                # Show trajectory (past positions)
                if frame_idx > 0:
                    t_past = t_frames[:frame_idx+1]
                    states_past = sol(t_past)
                    ax.plot(states_past[0], states_past[2], 'b--', alpha=0.5)

                pbar.update(1)

            # Create animation
            anim = FuncAnimation(fig, animate, frames=num_frames)

            # Save animation
            writer = FFMpegWriter(fps=30)
            anim.save(filename, writer=writer)

            pbar.close()
            plt.close(fig)
            print(f"Animation saved as {filename}")

            return filename

        # Create videos for all scenarios
        videos = []
        videos.append(create_video("scenario1_free_fall.mp4", f_phi_1, "Free Fall (f=0, phi=0)"))
        videos.append(create_video("scenario2_hovering.mp4", f_phi_2, "Hovering (f=Mg, phi=0)"))
        videos.append(create_video("scenario3_tilted.mp4", f_phi_3, "Tilted Thrust (f=Mg, phi=pi/8)"))
        videos.append(create_video("scenario4_landing.mp4", f_phi_4, "Controlled Landing"))

        return videos

    def create_snapshots():
        """
        Creates image snapshots of the booster location every 1 second for each scenario
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Common parameters
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]
        snapshot_times = np.linspace(0, 5, 6)  # 0, 1, 2, 3, 4, 5 seconds

        # Define the four scenarios
        scenarios = [
            {"name": "Free Fall (f=0, phi=0)", 
             "f_phi": lambda t, y: np.array([0.0, 0.0])},
            {"name": "Hovering (f=Mg, phi=0)", 
             "f_phi": lambda t, y: np.array([M*g, 0.0])},
            {"name": "Tilted Thrust (f=Mg, phi=pi/8)", 
             "f_phi": lambda t, y: np.array([M*g, np.pi/8])},
            {"name": "Controlled Landing", 
             "f_phi": lambda t, y: np.array([M * (6*a*t + 2*b + g), 0.0])}
        ]

        # Calculate coefficients for controlled landing
        y_final = l
        a = (2*(y0[2] - y_final) + t_span[1]*(y0[3])) / (t_span[1]**3)
        b = (3*(y_final - y0[2]) - t_span[1]*(2*y0[3])) / (t_span[1]**2)

        # Create a figure for each scenario
        for scenario in scenarios:
            # Solve trajectory
            sol = redstart_solve(t_span, y0, scenario["f_phi"])

            # Create figure with subplots for each time point
            fig = plt.figure(figsize=(15, 4))
            fig.suptitle(f"{scenario['name']} - Snapshots Every Second", fontsize=16)

            # Use GridSpec for better control of subplot layout
            gs = GridSpec(1, len(snapshot_times), figure=fig)

            # Calculate trajectory bounds for setting plot limits
            t_eval = np.linspace(t_span[0], t_span[1], 200)
            states = sol(t_eval)
            x_min, x_max = min(states[0]) - 2*l, max(states[0]) + 2*l
            y_min, y_max = min(min(states[2]) - 2*l, -1), max(states[2]) + 2*l

            # Ensure reasonable bounds
            x_min, x_max = min(x_min, -5), max(x_max, 5)
            y_min, y_max = min(y_min, -5), max(y_max, 15)

            for i, t in enumerate(snapshot_times):
                ax = fig.add_subplot(gs[0, i])
                ax.set_aspect('equal')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_title(f"t={t:.1f}s")
                ax.grid(True)

                # Get state
                state = sol(t)
                x, dx, y_pos, dy, theta, dtheta = state

                # Get force and phi
                f, phi = scenario["f_phi"](t, state)

                # Draw booster
                draw_rocket(x, y_pos, theta, f, phi, ax=ax)

                # Plot ground line
                ax.axhline(y=0, color='green', linestyle='-', alpha=0.3)

                # Only add x/y labels to leftmost/bottom plots to save space
                if i == 0:
                    ax.set_ylabel('y (height)')
                if i == len(snapshot_times) // 2:
                    ax.set_xlabel('x (position)')

            plt.tight_layout()
            # Create a safe filename by removing special characters
            safe_name = scenario['name'].split('(')[0].strip().replace(' ', '_').lower()
            filename = f"{safe_name}_snapshots.png"
            plt.savefig(filename)
            plt.close(fig)
            print(f"Created snapshots: {filename}")

        return "Snapshots created for all scenarios"

    # Run both visualization functions
    snapshots_result = create_snapshots()
    videos_result = create_booster_videos()

    print(f"Results:\n{snapshots_result}\n{videos_result}")
    return


if __name__ == "__main__":
    app.run()
