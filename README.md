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

## How to use

In order to use the present program, the user needs to run the script [integration](https://github.com/robertabenincasa/project_Lorenz/blob/master/integration.py) to obtain the numerical results of the analysis and the script [visualization](https://github.com/robertabenincasa/project_Lorenz/blob/master/visualization.py) to produce the graphical representation of the same data. All the results are saved in the folders [/output/data](https://github.com/robertabenincasa/project_Lorenz/blob/master/output/data) and [/output/plots](https://github.com/robertabenincasa/project_Lorenz/blob/master/output/plots). Moreover, a default configuration file, namely [config.ini](https://github.com/robertabenincasa/project_Lorenz/blob/master/config.ini), is available, but the user is allowed to use the file of their choice. However, it is requested that the new configuration file has the same sections and parameters of the default one. Since the script [config.py](https://github.com/robertabenincasa/project_Lorenz/blob/master/config.py) generates the configuration file with the desired structure, if one wants to generate a different file they could just change the values of the parameters and the name of the output file in this script without modifying the overall layout of the configuration. If further modifications are needed, one has to change also the main code.
Finally, in order to proceed with the testing, the user needs to run the file [test](https://github.com/robertabenincasa/project_Lorenz/blob/master/test.py) with *pytest test.py*.

The **libraries** and **modules** needed to run the program and produce the desired plots are:
* *configparser*: necessary to create and then read the configuration file;
* *numpy*;
* *scipy.stats*;
* *scipy.integrate.odeint*;
* *scipy.optimize.curve_fit*;
* *pandas*;
* *os.path*;
* *typing*;
* *matplotlib.plot*;
* *matplotlib.animation*;
* *dataframe_image*;
* *tabulate*;
* *Pillow* or an equivalent library.

For testing, the following are requested instead:
* *pytest*;
* *hypothesis*;
* *unittest*.

## The structure of the program

Six different scripts have been implemented in order to perform all the tasks: [config](https://github.com/robertabenincasa/project_Lorenz/blob/master/config.py), [integration](https://github.com/robertabenincasa/project_Lorenz/blob/master/integration.py), [visualization](https://github.com/robertabenincasa/project_Lorenz/blob/master/visualization.py), [lorenz](https://github.com/robertabenincasa/project_Lorenz/blob/master/lorenz.py) and [plots](https://github.com/robertabenincasa/project_Lorenz/blob/master/plots.py).

### config.py

As previously said, the file [config](https://github.com/robertabenincasa/project_Lorenz/blob/master/config.py) produces the configuration file to be used in the main code. The parameters whose value is defined in it are:
* *num_steps*: number of steps for the time integration of the Lorenz system;
* *dt*: width of the time step for the integration;
* *N*: number of ensemble members;
* *Random seed*: the random seed for the generation of an ensemble of random perturbation is fixed in the configuration file;
* *IC*: initial condition for the integration;
* *sigma*, *b*, *r1*, *r2*: the parameters of the Lorenz system, where *r* is set equal to 2 possible values, namely *r1* and *r2*, which corresponds to a chaotic  and a non-chaotic solution, respectively.
* *which_variable*: it indicates which variable among x, y and z of the initial condition *IC* is chosen to be perturbed;
* *eps*: vector of the applied perturbations to the initial condition;
* *threshold*: threshold to be used in the analysis to determine the predictability time of the system, i.e. when the RMSE becomes greater than this value.
* *which_eps_for_difference*,*which_eps_for_animation*: they are used in the plotting process in order to select which solution one wants to visualize.

The following paths are also specified there:
* *path_data*: location where the data from the numerical analysis are supposed to be stored;
* *path_plots*: location where the produced plots are saved.

### integration.py

The main code is [integration](https://github.com/robertabenincasa/project_Lorenz/blob/master/integration.py) in which the integration of the Lorenz system is performed for two different set of parameters in order to obtain a chaotic and a non-chaotic solution, respectively:
* set A: $\sigma$, *b*, *r1* = (10, 8/3, 28)
* set B: $\sigma$, *b*, *r2* = (10, 8/3, 9)

In order to further investigate the behaviour of the system, a perturbation on the initial condition is introduced and an analysis of the predictability of the system can be carried out. Assuming the unperturbed solution to be the *true* one, it is possible to compute the difference between the unperturbed and the perturbed one and to define the *Root Mean Square Error* as:

$$
\begin{equation}
RMSE = \sqrt{\frac{1}{3}\sum_{i=1}^{3}(x_{i}^{true}-x_{i}^{per})^{2}}
\end{equation}
$$

Note that the analysis, starting from the computation of the RMSE, is performed only on the chaotic solution, since it would have been uninformative for set B of parameters.
Moreover, in a chaotic system, the distance between the two trajectories $\delta$(t) grows as $\delta(t)\sim\delta_{0}\exp{\lambda t}$, where $\lambda$ is the maximum Lyapunov exponent (approximately 0.9 for the Lorenz system), so the predictability time $t\sim\frac{1}{\lambda}ln(\frac{a}{\delta_{0}})$ is supposed to decrease with increasing initial distance following a logarithmic relation. The predictability time is here arbitrarly calculated as the time at which the RMSE became greater than a certain threshold, defined in the configuration file, and is related to the sensitiveness to the initial condition typical of chaotic systems. For the sake of completeness, a linear fit of the predictability time as a function of the logarithm of the perturbation is performed in order to check what expected from the theory. However, since this relation is supposed to be valid for infinitesimal perturbations, 2 fits are performed and compared: one for perturnations up to $10_{-7}$ and the other for greater values of $\epsilon$. 

Finally, in order to show how the predictability of such a chaotic system can be improved, ensemble forecasting is considered. An ensemble of *N* random generated perturbations is applied as before and the Lorenz system is integrated for each IC. The ensemble mean and the ensemble spread are computed for each variable. Then, the RMSE of the ensemble mean (*L*) and the mean RMSE (*R*) are computed as a function of time:

$$
\begin{align}
L = \sqrt{\frac{1}{3}\sum_{i=1}^{3}(x_{i}^{true}-x_{i}^{ave})^{2}}   &&    R = \frac{1}{N}\sum_{j=1}^{N}(\sqrt{\frac{1}{3}\sum_{i=1}^{3}(x_{i}^{true}-x_{i,j}^{per})^{2}})
\end{align}
$$

and the associated predictability times are calculated. 
Ultimately, all the numerical results are saved to files and stored in the folder [/output/data](https://github.com/robertabenincasa/project_Lorenz/blob/master/output/data).

### lorenz.py

In the [lorenz](https://github.com/robertabenincasa/project_Lorenz/blob/master/lorenz.py) file all the functions used in the main code are defined:
* *reading_configuration_file*: allows a command line interface with the user in order to let them choose the configuration file that they want to use for the simulation. If none is given, the default one is used; 
* *lorenz*: returns the time derivative of the 3 variables x, y and z as given by the Lorenz system;
* *perturbation*: adds a perturbation to the chosen component of the initial condition of the simulation;
* *integration_Lorenz_system*: performs the integration of the Lorenz system, defined in the function lorenz, using the scipy.integrate.odeint function;
* *difference*: performs the difference between the  chosen components of 2 trajectories of the system;
* *RMSE*: performs the calculation of the root mean square error of the solution obtained from the perturbed ICs with respect to the unperturbed one;
* *generate_random_perturbation*: returns an array of N random numbers in the range between -0.75 and 0.75;
* *calculating_L_and_R*: calculates the mean of the RMSE of each members of the ensemble (R) and the RMSE of the ensemble mean (L);
* *ensemble*: performs the calculation of the ensemble mean and of the ensemble spread;
* *prediction*: finds the value of the predictability time for each value of the perturbation applied to the system;
* *func*: produces a linear equation( y = ax + b) necessary for the linear fitting;
* *fitting*: produces a linear fit of the predictability time as a function of the logarithm of the perturbation. 

Further information are available as docstrings in the program itself and can be accessed by typing *help('name of the function')*.

### plots.py and visualization.py

In the [plots](https://github.com/robertabenincasa/project_Lorenz/blob/master/plots.py) file all the functions necessary to plot the results are defined:
* *xzgraph*: produces a plot of the solution of the integration of the Lorenz system in the plane x, z. 
* *plot_3dsolution*: produces a 3D plot of the solution of the integration of the Lorenz system.
* *plot_animation*: produces an animation of the solution of the integration of the Lorenz system. Note that, in order to produce the animation, a suitable library is necessary, such as Pillow.
* *plot_difference*: produces a plot of the difference as a function of time;
* *plot_rmse*: produces a plot of the RMSE as a function of time;
* *plot_ensemble_trajectories*: produces a plot of the ensemble mean and its corresponding ensemble spread for each of the 3 variables;
* *plot_ensemble*: produces a plot of *L* and *R* as a function of time;
* *pred_time_vs_perturbation*: produces a plot of the predictability time as a function of the applied perturbation.

Further information are available as docstrings in the program itself and can be accessed by typing *help('name of the function')*.

These functions are imported in the script [visualization](https://github.com/robertabenincasa/project_Lorenz/blob/master/visualization.py) together with the data previousy produced. Then, the graphs are created, shown to the user and saved to the folder [/output/plots](https://github.com/robertabenincasa/project_Lorenz/blob/master/output/plots). Together with the plots, tables containing the predictability times with their corresponding applied perturbation are equally created, printed to terminal and saved as .png files.

### test.py

Finally, in the [test](https://github.com/robertabenincasa/project_Lorenz/blob/master/test.py) file all the functions defined in the [lorenz](https://github.com/robertabenincasa/project_Lorenz/blob/master/lorenz.py) file are tested through hypothesis testing in order to verify their proper functioning.
The property tested in each test is explained as a docstring in the file [test](https://github.com/robertabenincasa/project_Lorenz/blob/master/test.py) itself.

## Expected results

### Canonical system

The 2 systems, resulting from the integration using the 2 set of parameters, show completely different behaviours. As expected, for set B of parameters, the solution converges to a single point attractor, whereas for set A of parameters, the system exhibits chaotic behaviour, i.e. of a strange attractor. Any two arbitrarily close alternative initial points on the attractor, after any of various numbers of iterations, will lead to points that are arbitrarily far apart, but still subject to the confines of the attractor, and after any of various other numbers of iterations will lead to points that are arbitrarily close together. Thus, a dynamic system with a chaotic attractor is locally unstable yet globally stable: once in the attractor, nearby points diverge from one another but never depart from the attractor. 
Moreover, it is immediate to show the dissipative nature of the system in the second case, since it converges to a single point. Instead, in the former scenario this condition is satisfied because a strange attractor has a fractal structure, which has zero volume in phase space.

### Single perturbation of the initial condition

For set B, the 2 trajectories relaxes to a single one after a brief oscillating transient and their difference tends to zero accordingly. Instead, for set A of parameters, the 2 trajectories suddenly distance each other after a transient in which they coincide and start oscillating independently in a chaotic manner and so does their difference. The RMSE increases in an exponential manner and saturates at the size a of the attractor. That means that the RMSE, i.e. the distance between the 2 trajectories, cannot be greater than the dimension of the attractor itself, since they are confined to it. 

### Ensemble of perturbations

It is possible to show that the RMSE of the ensemble mean is clearly smaller than the mean RMSE or the RMSE of any simulation and the corresponding predictability times for *L* and *R* show how the predictability time window is expanded in the first case. 


Here the resulting animation of the Lorenz system:
![](https://github.com/robertabenincasa/project_Lorenz/blob/master/output/plots/animation.gif)

