# requires only numpy
import numpy as np

class SDEBE(object):
    ''' Numerically solve an n-dimensional stochastic system of the form:
            dx = f(x, t)dt + G(x, t)dW
        Currently supports the backward (implicit) Euler method, using Newton's
        method to iteratively approximate the roots of f to solve the numerical
        scheme:
            x(n+1) ~ x(n) + f(x(n+1), t)dt + G(x(n), t)dW

        Parameters :
            - f (required) :
                a function with inputs y (current state) and t (current time)
                outputting an n-length array for the diffusion at that state
                and time
            - g (required) :
                a function with inputs y (current state) and t (current time)
                outputting an n-by-n dimensional matrix for the drift at the
                current state
            - jac (required) :
                a function with inputs y (crurent state) and t (current time)
                outputting an n-by-n dimensional matrix for the jacobian of
                the diffusion at the current state and time, Df(y, t)

        Methods:
            - integrate
                Inputs:
                    - t0 : initial time (scalar)
                    - y0 : n-length initial state vector
                    - dt : time step
                    - tmax | periods : tmax is the maximum time, periods is
                                       the times to iterate. either tmax or
                                       periods is required
                    - (maxiter) : an integer for the maximum number of times
                                  to iterate when using Newton's method to
                                  find the roots of the nonlinear system
                integrate numerically solves the sde, saving the states to
                variables sde.ys and sde.ts
            - plot
                Inputs:
                    - labels : n-length string list, with the i-th component
                        describing the i-th state variable
                    - fname : the output filename
                Saves a figure `fname' with labels `labels' to the current
                directory, a scatterplot of each state variable
                '''
    def __init__(self, f=None, g=None, jac=None):
        ''' Initialize system with f, g, and jacobian functions '''
        if not f or not g or not jac:
            raise TypeError('SDEBE requires f, g, and jacobian functions')
        self.f = f
        self.g = g
        self.jac = jac

    def integrate(self, t0, y0, dt, tmax=None, periods=None, maxiter=12):
        ''' Iterate using Newton's method and initial conditions '''
        self.dt = dt
        if tmax:
            periods = int(tmax / dt)
        elif periods:
            tmax = dt * periods
        else:
            e = 'SDEBE.integrate requires either tmax or periods kwarg'
            raise TypeError(e)

        # initialize
        self.ts = np.linspace(t0, tmax, periods)
        self.ys = np.zeros((len(y0), len(self.ts)))
        yn = y0
        # for progress update
        niters = 0
        every = 10000
        print '0.00%'
        for i, tn in enumerate(self.ts):
            if every:
                every -= 1
            else:
                every = 10000
                print '%s%%' % str(round(float(niters) / periods * 100,2))

            ynp1_approx, fn = self.newton_approx(yn, tn, maxiter)
            drift = fn * dt
            diffusion = np.dot(self.g(yn, tn),
                               np.random.normal(scale=np.sqrt(dt),
                                                size=len(y0)))
            ynp1 = yn + drift# + diffusion
            self.ys[:, i] = ynp1
            self.ts[i] = tn
            yn = ynp1
            tn += dt
            niters += 1

        print '100%'

    def newton_approx(self, y, t, iterations):
        Df = self.jac(y, t)
        Dfinv = np.linalg.inv(Df)
        while iterations:
            try:
                fn = self.f(y, t)
                y_new = y - np.dot(Dfinv, fn)
                if np.linalg.norm(y_new - y) < 1e-11:
                    return y, fn
                else:
                    y = y_new

            except np.linalg.linalg.LinAlgError:
                print iterations
                raise TypeError
            iterations -= 1

        print 'Iterations --> 0'
        return y, fn

    def plot(self, labels, fname, save=False):
        import matplotlib.pyplot as plt
        if len(labels) != len(self.ys):
            raise ValueError('Must have as many labels as variables')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i in range(len(labels)):
            print 'Plotting %s...' % labels[i]
            plt.plot(self.ts, self.ys[i, :], color=colors[i])

        plt.legend(labels)
        if save:
            plt.savefig(fname, dpi=200)
            print 'Figure saved to "%s".' % fname
        else:
            plt.show()

if __name__ == '__main__':
    print 'functions.py file being used'
    from functions import *
    # print out a breakdown of how long each method takes to run
    profile = True
    # save output variable to a json
    save_data = True
    # save plots
    save_plots = False

    periods = 7.3e7
    time_step = 2e-6

    if not profile:
        print 'Initializing system...'
        sde = SDEBE(f, G, jac)
        print 'Integrating...'
        sde.integrate(0, np.zeros(5), time_step, periods=periods)
        print 'Generating plots...'
    else:
        import cProfile, pstats, StringIO
        pr = cProfile.Profile()
        s = StringIO.StringIO()
        sortby = 'cumulative'

        sde = SDEBE(f, G, jac)

        pr.enable()
        sde.integrate(0, np.zeros(5), 0.0001, periods=periods)
        pr.disable()

        
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

    labels = ['x', 'y', 'v', 'T', 'S']
    sde.plot(labels, 'stochastic.png', save=save_plots)

    if save_data:
        print 'Saving data...'
        import json
        data = {}
        data['t'] = sde.ts.tolist()
        data['y'] = sde.ys.tolist()
        json.dump(data, open('stochastic.json', 'w'))

    print 'Done.'
