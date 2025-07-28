import numpy as np
import scipy.sparse as spar
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


k = 0.02
mu = 3e-3
rho = 2
beta = 0.6


Lx = 1
Nx = 100
dx = Lx/Nx

xvals = np.linspace( 0, Lx, Nx+1 )

Ly = 1
Ny = 100
dy =  Ly/Ny

yvals = np.linspace( 0, Ly, Ny+1 )

X,Y = np.meshgrid( xvals, yvals, indexing='ij' )


dt = 0.01
tmax = 10
tvals = np.arange( 0, tmax + dt, step=dt )


skip = ( slice(None, None, 2), slice(None, None, 2) )


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



# -- fixed value ceiling and insulated floor

Py_lid = spar.diags(
    [ [-1]+[0]*(Ny-2), ones_y[1:], - ones_y[1:] ]
    , [0,1,-1]
    ) * 0.5/dy

Pyy_lid = spar.diags(
    [ [-1]+[-2]*(Ny-2), ones_y[1:], ones_y[1:] ]
    , [0,1,-1]
    ) * 1/dy**2

Dy_lid, Dyy_lid = spar.kron( Ix, Py_lid ), spar.kron( Ix, Pyy_lid )

Lap_lid = Dxx_ins + Dyy_lid



# -----------------------------------------------------------------------------
# -- Methods


def flow_step( u, v, f, b, S, k, rho ):
    uu, vv = spar.diags( [u], [0] ), spar.diags( [v], [0] )
    mat = Itens - dt * ( - uu @ Dx - vv @ Dy + k/rho * Lap )
    bound_vec = (
        - u * ( b[0] + b[1] )
        - v * ( b[2] + b[3] )
        + k/rho * sum( b[i] for i in [4,5,6,7] )
        )
    vec = f + dt * ( bound_vec + S )
    return spar.linalg.spsolve( mat, vec )

def fluid_momentum_step( u, v, ub, vb, Sx, Sy, mu, rho ):
    u1 = flow_step( u, v, u, ub, Sx, mu, rho )
    v1 = flow_step( u, v, v, vb, Sy, mu, rho )
    return u1, v1

def heat_step( u, v, Q, Sq, k, rho ):
    uu, vv = spar.diags( [u], [0] ), spar.diags( [v], [0] )
    mat = Itens - dt * ( - uu @ Dx - vv @ Dy + k/rho * Lap_ins )
    vec = Q + dt * Sq
    return spar.linalg.spsolve( mat, vec )


# -- Projection onto divergence free VFs


# Lap_ext = np.ones( ( (Nx-1)*(Ny-1) + 1, )*2 )
# Lap_ext[:-1, :-1] = Lap_ins.toarray()
# Lap_ext[-1, -1] = 0
# Lap_ext = spar.csr_matrix( Lap_ext )


# def Poisson_solver_new( S ):
#     S_ext = np.append(S, 0)
#     out = spar.linalg.spsolve( Lap_ext, S_ext )
#     return out[:-1]


def Poisson_solver_lid( S ):
    return spar.linalg.spsolve( Lap_lid, S )



def Chorin( u, v, ub, vb ):
    div = Dx @ u + Dy @ v + ub[0] + ub[1] + vb[2] + vb[3]
    p = Poisson_solver_lid( div )
    u1, v1 = u - Dx_ins @ p, v - Dy_lid @ p
    return u1, v1


# -- Combined fluid and heat stepper

def fluid_step( u, v, Q, ub, vb, Sx, Sy, Sq, mu, k, rho ):
    u0, v0 = fluid_momentum_step( u, v, ub, vb, Sx, Sy, mu, rho )
    Q1 = heat_step( u, v, Q, Sq, k, rho )
    u1, v1 = Chorin( u0, v0, ub, vb )
    return u1, v1, Q1




# -- Return to matrix form

def unflatten(f, f0):
    mat = np.zeros_like(X)
    mat[1:-1, 1:-1] = f.reshape( np.shape(X[1:-1, 1:-1]) )
    mat[0], mat[-1] = f0[0], f0[-1]
    mat[:,0], mat[:,-1] = f0[:,0], f0[:,-1]
    return mat

def unflatten_ins(f):
    mat = np.zeros_like( X )
    mat[1:-1, 1:-1] = f.reshape( np.shape( X[1:-1, 1:-1] ) )
    mat[0], mat[-1] = mat[1], mat[-2]
    mat[:,0], mat[:,-1] = mat[:,1], mat[:,-2]
    return mat



# -- Boundary Terms

def boundary_dir( f0 ):
    mat = []
    for i in range(8):
        mat.append( np.zeros_like(X[1:-1, 1:-1]) )
    # -- x-boundaries
    mat[0][0], mat[4][0] = -0.5/dx * f0[0, 1:-1], 1/dx**2 * f0[0, 1:-1]
    mat[1][-1], mat[5][-1] = 0.5/dx * f0[-1, 1:-1], 1/dx**2 * f0[-1, 1:-1]
    # -- y-boundaries
    mat[2][:,0], mat[6][:,0] = -0.5/dy * f0[1:-1, 0], 1/dy**2 * f0[1:-1, 0]
    mat[3][:,-1], mat[7][:,-1] = 0.5/dy * f0[1:-1, -1], 1/dy**2 * f0[1:-1, -1]
    return [ mat[i].flatten() for i in range(8) ]


# -----------------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------------

funx = (
        - np.cos( 2 * np.pi * X/Lx )
        * np.heaviside( X - Lx/4, 0.5 )
        * np.heaviside( 3*Lx/4 - X, 0.5 )
        )
funy = np.sin( 2*np.pi*Y/Ly ) * np.heaviside( Ly/2 - Y, 0.5 )

gen_fun = 0 * (funx * funy)[1:-1, 1:-1].flatten()

# -------------------  /|\
# -- Testing Purposes   |


u0 = np.zeros_like(X)
u0[:,-1] = 1

v0 = np.zeros_like(X)
u,v = u0[1:-1, 1:-1].flatten(), v0[1:-1, 1:-1].flatten()

u_bnd, v_bnd = boundary_dir(u0), boundary_dir(v0)


# Q0 = 0.5*( 1 - np.cos( 2*np.pi * X/Lx ) ) * ( 1 - Y/Ly ) * np.exp( -2*X/Lx ) * 2.4

# Q0 = 1 - Y/Ly
Q = np.zeros_like(u)




# -- Source Terms
Sx, Sy = np.zeros_like(u), gen_fun - beta * ( Dy_ins @ Q )

# Sq = 0.6 * ( funx * funy )[1:-1, 1:-1].flatten()
Sq = np.zeros_like(u)



# -----------------------------------------------------------------------------
# Plot setup
# -----------------------------------------------------------------------------


zerogrid = np.zeros_like(X)

fig, ax = plt.subplots( figsize=[10,8], dpi=70 )

fig.suptitle( 'Flow Velocity for a Lid Driven Cavity (with speed)' )

ax.set_xlabel('x')
ax.set_ylabel('y')


heatmap = ax.pcolormesh( X, Y, zerogrid, cmap='jet', vmin=0, vmax=1.1 )
fig.colorbar( heatmap, ax=ax )

qplot = ax.quiver(X, Y, zerogrid, zerogrid, pivot = 'middle' )

plt.show()



# -----------------------------------------------------------------------------
# Animation and simulation
# -----------------------------------------------------------------------------


max_count = 0.1/dt
count = max_count

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
            
            u_mat, v_mat = unflatten(u, u0), unflatten(v, v0)
            Q_mat = unflatten_ins(Q)
            speed = np.sqrt( u_mat**2 + v_mat**2 )
            
            heatmap = ax.pcolormesh(X, Y, speed, cmap='jet', vmin=0, vmax=1.1 )
            qplot = ax.quiver( X[skip], Y[skip],
                              u_mat[skip], v_mat[skip],
                              pivot = 'middle' )
            
            writer.grab_frame()
        
        Sy = gen_fun - beta * ( Dy_ins @ Q )
        u,v,Q = fluid_step( u, v, Q, u_bnd, v_bnd, Sx, Sy, Sq, mu, k, rho )



