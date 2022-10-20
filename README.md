# Lorenz system

The aim of the following project is to perform a numerical integration of the Lorenz system and to study it in presence of perturbations.
The Lorenz system is a nonlinear, non-periodic, three-dimensional and deterministic system of ordinary differential equations. It was developed by Lorenz in 1963 as a simplified mathematical model for atmospheric convection, studying the case of a two-dimensional fluid layer uniformly warmed from below and cooled from above. In its canonical form it is written as:

$$
\begin{equation}
    \begin{cases}
    \dot x=\sigma (y-x) \\
    \dot y=rx-xz-y \\
    \dot z=xy-bz
    \end{cases}
\end{equation} 
$$

The equations describe the rate of change of three quantities with respect to time: *x* is proportional to the intensity of convection, *y* to the horizontal temperature variation, i.e. the temperature difference between the ascending and descending branches, and *z* is proportional to the distortion of the average vertical profile of temperature from linearity. The parameters $\sigma$ and *b* are related to the type of fluid and its properties and to the geometry of the problem, respectively. Instead, the parameter *r*, linked to the Rayleigh number, is a measure of the behaviour of the fluid and gives information about the presence or not of convection.

More generally, the Lorenz system is a canonical example of a system that shows chaotic behaviour for certain values of the parameters. It can be demonstrated that, assuming $\sigma$ and *b* to be constant and positive, the behaviour of the system depends on the parameter *r*. For 0 < *r* < 1, the system has one stable stationary point that coincides with the origin. However, at *r* = 1 a bifurcation occurs and 2 additional stationary points are born, while the origin becomes nonstable. Finally, for *r* > 24.74.. these two points become unstable too and the system is chaotic for *r* = 28. For completeness, it is important to notice that the Lorenz system is dissipative, so it means that the volume in phase space contracts with time to an attracting set of zero volume:

$$
\begin{equation}
    \frac{\partial \dot x}{\partial x}+\frac{\partial \dot y}{\partial y}+\frac{\partial \dot z}{\partial z}=-(\sigma+b+1)
\end{equation}
$$

For more information about the Lorenz system, see *Atmospheric modeling, data assimilation and predictability*, E. Kalney, 2003.


## Canonical system

First, the numerical integration of the Lorenz system is performed, with initial condition L<sub>0</sub> = (x<sub>0</sub>, y<sub>0</sub>, z<sub>0</sub>). Two different set of parameters were adopted in order to obtain a chaotic and a non-chaotic solution, respectively:
* set A: $\sigma$, *b*, *r* = (10, 8/3, 28)
* set B: $\sigma$, *b*, *r* = (10, 8/3, 9)

The 2 systems show completely different behaviours. As expected, for set B of parameters, the solution converges to a single point attractor, whereas for set A of parameters, the system exhibits chaotic behaviour, i.e. of a strange attractor. Any two arbitrarily close alternative initial points on the attractor, after any of various numbers of iterations, will lead to points that are arbitrarily far apart, but still subject to the confines of the attractor, and after any of various other numbers of iterations will lead to points that are arbitrarily close together. Thus, a dynamic system with a chaotic attractor is locally unstable yet globally stable: once in the attractor, nearby points diverge from one another but never depart from the attractor. 
Moreover, it is immediate to show the dissipative nature of the system in the second case, since it converges to a single point. Instead, in the former scenario this condition is satisfied because a strange attractor has a fractal structure, which has zero volume in phase space.

## Single perturbation of the initial condition

By introducing a perturbation on the initial condition of the type E = ( $\epsilon$ ,0,0), the behaviour of the system can be further investigated. Assuming the unperturbed solution to be the *true* one, it is possible to compute the difference between the unperturbed and the perturbed one and to define the *Root Mean Square Error* as:

$$
\begin{equation}
RMSE = \sqrt{\sum_{i=1}^{3}(x_{i}^{true}-x_{i}^{per})^{2}}
\end{equation}
$$

For set B, the 2 trajectories relaxes to a single one after a brief oscillating transient and their difference tends to zero accordingly. Instead, for set A of parameters, the 2 trajectories suddenly distance each other after a transient in which they coincide and start oscillating independently in a chaotic manner and so does their difference. The RMSE increases in an exponential manner and saturates at the size a of the attractor. That means that the RMSE, i.e. the distance between the 2 trajectories, cannot be greater than the dimension of the attractor itself, since they are confined to it. Moreover, in a chaotic system, the distance between the two trajectories $\delta$(t) grows as $\delta(t)\sim\delta_{0}\exp{\lambda t}$, where $\lambda$ is the maximum Lyapunov exponent (approximately 0.9 for the Lorenz system), so the predictability time $t\sim\frac{1}{\lambda}ln(\frac{a}{\delta_{0}})$ is supposed to decrease with increasing initial distance following a logarithmic relation. The predictability time is here arbitrarly defined as the time at which the RMSE became greater than 0.5 and is related to the sensitiveness to initial conditions typical of chaotic systems.

## Ensemble of perturbations
Finally, in order to show how the predictability of such a chaotic system can be improved, ensemble forecasting is considered. An ensemble of *N* random generated perturbations is applied as before and the Lorenz system is integrated for each IC. Then, the RMSE of the ensemble mean (*L*) and the mean RMSE (*R*) are computed as a function of time:

$$
\begin{align}
L = \sqrt{\sum_{i=1}^{3}(x_{i}^{true}-x_{i}^{ave})^{2}}   &&    R = \frac{1}{N}\sum_{j=1}^{N}(\sqrt{\sum_{i=1}^{3}(x_{i}^{true}-x_{i,j}^{per})^{2}})
\end{align}
$$

Consequently, it is possible to show that the RMSE of the ensemble mean is clearly smaller than the mean RMSE or the RMSE of any simulation. To be consistent, the corresponding predictability times for *L* and *R* are computed in order to show how the predictability time window is expanded. 
## The code
Four different scripts are used in order to perform all the tasks previously described.
First of all, the [configuration](https://github.com/robertabenincasa/project_Lorenz/blob/master/config.py) file must be compiled by the user in order to set the values of the integration parameters and to specify the local path to the repository where the output of the code is supposed to be saved. By running the configuration file, the [*config.ini*](https://github.com/robertabenincasa/project_Lorenz/blob/master/config.ini) is produced which it is then imported by the main code with the ConfigParser library. 

The parameters used in the simulation are:
* *num_steps*: the number of steps for the integration;
* *dt*: the step size;
* *N*: number of random perturbations;
* *b*,  $\sigma$, *r1*, *r2*: the values of the parameters of the Lorenz system as defined above;
* *IC*: the initial condition of the system;
* *eps*: the values of the perturbations applied to the system.

Their values can be modified by the users according to their needs, while keeping in mind the analytical description of the system provided before. 

In order to obtain the entire output, it is necessary to run only the [integration](https://github.com/robertabenincasa/project_Lorenz/blob/master/integration.py) script, which is also the main code of the project. The time integration of the Lorenz system is performed through the [scipy.integrate.odeint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html) library. Note that the integration is performed for each set of parameters and for each initial conditions. The latter are obtained by perturbing the x-component of the original initial condition *IC* through the perturbations defined in the array *eps*. Then, the difference between the x-component of the unperturbed trajectory and one of the perturbed are computed for both set of parameters, as a preliminary analysis. Subsequently, the Root Mean Square Error is computed for each value of the perturbation only for the chaotic solution, since it would have been uninformative for set B of parameters. Moreover, the predictability time is computed and stored in a table alongside with its corresponding value of $\epsilon$ and a linear fit of the predictability time as a function of the logarithm of the perturbation is performed in order to check what expected from the theory. 
Finally, the same procedure is repeated for the case of an ensemble of pertubations and the corresponding predictability times are stored in a second table.

In the [lorenz](https://github.com/robertabenincasa/project_Lorenz/blob/master/lorenz.py) file all the functions used in the main code are defined:
* *read_parameters*: converts a string composed of numbers separated by a comma into the corresponding np.array. It was realised in order to read the values of some parameters in the configuration file which are conceived to be vectors, but were written as strings.
* *lorenz*: returns the time derivative of the 3 variables x, y and z as given by the Lorenz system.
* *perturbation*: adds a perturbation to the first component of the initial condition of the simulation.
* *difference*: performs the difference between the x-components of 2 trajectories of the system.
* *RMSE*: performs the calculation of the root mean square error of the solution obtained from the perturbed ICs with respect to the unperturbed one.
* *ensemble*: performs the calculation of the ensemble mean and of the ensemble spread;
* *prediction*: finds the value of the predictability time for each value of the perturbation applied to the system;
* *func*: produces a linear equation( y = ax + b) necessary for the linear fitting;
* *fitting*: produces a linear fit of the predictability time as a function of the logarithm of the perturbation. 

Further information are available as docstrings in the program itself and can be accessed by typing *help('name of the function')*.

In the [plots](https://github.com/robertabenincasa/project_Lorenz/blob/master/plots.py) file all the functions necessary to plot the results are defined:
* *xzgraph*: produces a plot of the solution of the integration of the Lorenz system in the plane x, z. 
* *plot_3dsolution*: produces a 3D plot of the solution of the integration of the Lorenz system.
* *plot_animation*: produces an animation of the solution of the integration of the Lorenz system. Note that, in order to produce the animation, a suitable library is necessary, such as Pillow.
* *plot_difference*: produces a plot of the difference as a function of time.
* *plot_rmse*: produces a plot of the RMSE as a function of time;
* *plot_ensemble_trajectories*: produces a plot of the ensemble mean and its corresponding ensemble spread for each of the 3 variables;
* *plot_ensemble*: produces a plot of *L* and *R* as a function of time;
* *pred_time_vs_perturbation*: produces a plot of the predictability time as a function of the applied perturbation.

The graphs are automatically shown and saved in the repository [output](https://github.com/robertabenincasa/project_Lorenz/blob/master/output) by running the main code.

Finally, in the [test](https://github.com/robertabenincasa/project_Lorenz/blob/master/test.py) file all the functions defined in the [lorenz](https://github.com/robertabenincasa/project_Lorenz/blob/master/lorenz.py) file are tested through hypothesis testing in order to verify their proper functioning.

Here the resulting animation of the Lorenz system:
![](https://github.com/robertabenincasa/project_Lorenz/blob/master/output/animation.gif)

