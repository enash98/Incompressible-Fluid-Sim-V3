import numpy as np
import scipy.sparse as spar
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


k = 0.04
mu = 0.01
rho = 2
beta = 2


Lx = 1
Nx = 35
dx = Lx/Nx

xvals = np.linspace( 0, Lx, Nx+1 )

Ly = 1
Ny = 35
dy =  Ly/Ny

yvals = np.linspace( 0, Ly, Ny+1 )

X,Y = np.meshgrid( xvals, yvals, indexing='ij' )


dt = 0.01
tmax = 7.5
tvals = np.arange( 0, tmax + dt, step=dt )

# -----------------------------------------------------------------------------
# Sparse Matrices


# -- Fixed Boundary

ones_x = np.ones(Nx-1)
ones_y = np.ones(Ny-1)

Ix = spar.identity(Nx-1)
Iy = spar.identity(Ny-1)

Itens = spar.identity( (Nx-1)*(Ny-1) )

Px = spar.diags( [ ones_x[1:], -ones_x[1:] ], [1,-1] ) * 0.5/dx
Py = spar.diags( [ ones_y[1:], -ones_y[1:] ], [1,-1] ) * 0.5/dy

Pxx = spar.diags( [ -2*ones_x, ones_x[1:], ones_x[1:] ], [0,1,-1] ) / dx**2

Pyy = spar.diags( [ -2*ones_y, ones_y[1:], ones_y[1:] ], [0,1,-1] ) / dy**2

Dx, Dy = spar.kron( Px, Iy ), spar.kron( Ix, Py )
Dxx, Dyy = spar.kron( Pxx, Iy ), spar.kron( Ix, Pyy )

Lap = Dxx + Dyy



# -- Insulated boundary

Px_ins = spar.diags(
    [ [-1]+[0]*(Nx-3)+[1], ones_x[1:], - ones_x[1:] ]
    , [0,1,-1]
    ) * 0.5/dx

Py_ins = spar.diags(
    [ [-1]+[0]*(Ny-3)+[1], ones_y[1:], - ones_y[1:] ]
    , [0,1,-1]
    ) * 0.5/dy

Pxx_ins = spar.diags(
    [ [-1]+[-2]*(Nx-3)+[-1], ones_x[1:], ones_x[1:] ]
    , [0,1,-1]
    ) * 1/dx**2

Pyy_ins = spar.diags(
    [ [-1]+[-2]*(Ny-3)+[-1], ones_y[1:], ones_y[1:] ]
    , [0,1,-1]
    ) * 1/dy**2

Dx_ins, Dy_ins = spar.kron( Px_ins, Iy ), spar.kron( Ix, Py_ins )
Dxx_ins, Dyy_ins = spar.kron( Pxx_ins, Iy ), spar.kron( Ix, Pyy_ins )

Lap_ins = Dxx_ins + Dyy_ins



# -----------------------------------------------------------------------------
# -- Methods


def flow_step( u, v, f, S, k, rho ):
    uu, vv = spar.diags( [u], [0] ), spar.diags( [v], [0] )
    mat = Itens - dt * ( - uu @ Dx - vv @ Dy + k/rho * Lap )
    vec = f + dt * S
    return spar.linalg.spsolve( mat, vec )

def fluid_momentum_step( u, v, Sx, Sy, mu, rho ):
    u1 = flow_step( u, v, u, Sx, mu, rho )
    v1 = flow_step( u, v, v, Sy, mu, rho )
    return u1, v1

def heat_step( u, v, Q, Sq, k, rho ):
    uu, vv = spar.diags( [u], [0] ), spar.diags( [v], [0] )
    mat = Itens - dt * ( - uu @ Dx - vv @ Dy + k/rho * Lap_ins )
    vec = Q + dt * Sq
    return spar.linalg.spsolve( mat, vec )



# -- Projection

h = 1e-3
PS_mat = Itens - h * Lap_ins

def Poisson_solver( f0, S, steps ):
    f = f0
    for i in range(steps):
        f = spar.linalg.spsolve( PS_mat, f - h * S )
    
    return f

PS_steps = 25

def Chorin( u, v ):
    div = Dx @ u + Dy @ v
    p0 = np.zeros_like(u)
    p = Poisson_solver( p0, div, PS_steps )
    
    u1, v1 = u - Dx_ins @ p, v - Dy_ins @ p
    
    return u1, v1


# -- Combined fluid and heat stepper

def fluid_step( u, v, Q, Sx, Sy, Sq, mu, k, rho ):
    u0, v0 = fluid_momentum_step( u, v, Sx, Sy, mu, rho )
    Q1 = heat_step( u, v, Q, Sq, k, rho )
    u1, v1 = Chorin( u0, v0 )
    return u1, v1, Q1





# -- Return to matrix form

def unflatten(f):
    mat = np.zeros_like(X)
    mat[1:-1, 1:-1] = f.reshape( np.shape(X[1:-1, 1:-1]) )
    return mat

def unflatten_ins(f):
    mat = np.zeros_like(X)
    mat[1:-1, 1:-1] = f.reshape( np.shape(X[1:-1, 1:-1]) )
    mat[0], mat[-1] = mat[1], mat[-2]
    mat[:,0], mat[:,-1] = mat[:,1], mat[:,-2]
    
    return mat



# -----------------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------------


funx = (
        - np.cos( 2 * np.pi * X/Lx )
        * np.heaviside( X - Lx/4, 0.5 )
        * np.heaviside( 3*Lx/4 - X, 0.5 )
        )
funy = np.sin( 2*np.pi*Y/Ly ) * np.heaviside( Ly/2 - Y, 0.5 )


u0 = v0 = np.zeros_like(X)
u,v = u0[1:-1, 1:-1].flatten(), v0[1:-1, 1:-1].flatten()

Q = np.ones_like(u)

# Sy0 = 4 * ( funx * funy )[1:-1, 1:-1].flatten()
Sx, Sy = np.zeros_like(u), - beta * Dy_ins @ Q


Sq = 0.75 * ( funx * funy )[1:-1, 1:-1].flatten()



# -----------------------------------------------------------------------------
# Plot setup
# -----------------------------------------------------------------------------


zerogrid = np.zeros_like(X)

fig, ax = plt.subplots( figsize=[10,8], dpi=80 )

fig.suptitle( 'Plot of flow velocity and heat distribution' )

ax.set_xlabel('x')
ax.set_ylabel('y')


heatmap = ax.pcolormesh( X, Y, zerogrid, vmin=1, vmax=2 )
fig.colorbar( heatmap, ax=ax )

qplot = ax.quiver(X, Y, zerogrid, zerogrid, pivot = 'middle' )

plt.show()



# -----------------------------------------------------------------------------
# Animation and simulation
# -----------------------------------------------------------------------------


# maxstep = 200
max_count = 0.1/dt
count = max_count-1

MD = dict( title = '', artist = '' )

writer = PillowWriter( fps = 10, metadata = MD )

with writer.saving( fig, 'New Heat Plot.gif', 10*tmax+1 ):
    for t in tvals:
        
        if count+1 < max_count:
            count += 1
        else:
            count = 0
            
            qplot.remove()
            heatmap.remove()
            
            u_mat, v_mat, Q_mat = unflatten(u), unflatten(v), unflatten_ins(Q)
            # speed = np.sqrt( u_mat**2 + v_mat**2 )
            
            heatmap = ax.pcolormesh(X, Y, Q_mat, vmin=1, vmax=2)
            qplot = ax.quiver(X, Y, u_mat, v_mat, pivot = 'middle' )
            
            writer.grab_frame()
        
        Sy = - beta * Dy_ins @ Q
        u,v,Q = fluid_step( u, v, Q, Sx, Sy, Sq, mu, k, rho )



