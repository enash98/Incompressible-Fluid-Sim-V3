import numpy as np
import scipy.sparse as spar
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


rho = 2
mu = 5e-3

skip = ( slice(None, None, 2), slice(None, None, 2) )


# -----------------------------------------------------------------------------
# Grid Setup

r_inn = 0.5
r_out = 2
Lr = r_out - r_inn
Nr = 60
dr = Lr/Nr

rvals = np.linspace( r_inn, r_out, Nr+1 )

Na = 120
da = 2*np.pi / Na

avals = np.linspace( 0, 2*np.pi, Na+1 )

R,A = np.meshgrid( rvals, avals, indexing='ij' )

# Cartesian Coordinates
X,Y = R * np.cos(A), R * np.sin(A)


tmax = 12
dt = 0.01
tvals = np.arange( 0, tmax, step=dt )


# -----------------------------------------------------------------------------
# Sparse matrices

Ir = spar.identity(Nr-1)
Ia = spar.identity(Na-1)

Iprod = spar.identity( (Nr-1)*(Na-1) )

ones_r = np.ones( Nr-1 )
ones_a = np.ones( Na-1 )


# -- Radial Derivative Operators (Fixed end values)

Pr = spar.diags( [ ones_r[1:], -ones_r[1:] ], [1,-1] ) * 0.5/dr

Prr = spar.diags( [ -2*ones_r, ones_r[1:], ones_r[1:] ], [0,1,-1] ) * 1/dr**2

Dr, Drr = spar.kron( Pr, Ia ), spar.kron( Prr, Ia )


# -- Radial Derivative Operators (Inuslated ends)

Pr_ins = spar.diags(
    [  [-1] + [0]*(Nr-3 ) + [1], ones_r[1:], -ones_r[1:]  ],
    [0,1,-1]
    ) * 0.5/dr

Prr_ins = spar.diags(
    [  [-1] + [-2]*(Nr-3 ) + [-1], ones_r[1:], ones_r[1:]  ],
    [0,1,-1]
    ) * 1/dr**2

Dr_ins, Drr_ins = spar.kron( Pr_ins, Ia ), spar.kron( Prr_ins, Ia )


# -- Angular Derivative Operators

corns_top = np.zeros( (Na-1, )*2 )
corns_bot = np.zeros( (Na-1, )*2 )
corns_top[0] = corns_bot[-1] = [0.5] + [0]*(Na-3) + [0.5]
corns_top_sp = spar.csr_matrix( corns_top )
corns_bot_sp = spar.csr_matrix( corns_bot )

Pa = (
          spar.diags( [ ones_a[1:], -ones_a[:1] ], [1,-1] )
          - corns_top_sp
          + corns_bot_sp
      ) * 0.5/da

Paa = (
           spar.diags( [ -2*ones_a, ones_a[1:], ones_a[1:] ], [0,1,-1] )
           + corns_top_sp
           + corns_bot_sp
       ) * 1/da**2


Da, Daa = spar.kron(Ir, Pa), spar.kron( Ir, Paa )


# -- Radial Coordinate array

rcoo = R[1:-1, 1:-1].flatten()
acoo = A[1:-1, 1:-1].flatten()

rinv_sp = spar.diags( [ 1/rcoo ], [0] )
rinv2_sp = spar.diags( [ 1/rcoo**2 ], [0] )


# -- Laplacian Matrices

Lap_mat = rinv_sp @ Dr + Drr + rinv2_sp @ Daa

Lap_ins = rinv_sp @ Dr_ins + Drr_ins + rinv2_sp @ Daa


# -----------------------------------------------------------------------------




def fluid_momentum_step( u, v, ub, vb, Sr, Sa, mu, rho ):
    uu, vv = spar.diags( [u], [0] ), spar.diags( [v], [0] )
    mat = - uu @ Dr - vv @ rinv_sp @ Da + mu/rho * Lap_mat
    op_mat = Iprod - dt * mat
    
    bvec_r = (
        - u * ( ub[0] + ub[1] )
        + mu/rho * (  ( ub[0] + ub[1] ) / rcoo + ub[2] + ub[3]  )
        )
    
    bvec_a = (
        - u * ( vb[0] + vb[1] )
        + mu/rho * (  ( vb[0] + vb[1] ) / rcoo + vb[2] + vb[3]  )
        )
    
    vec_r = u + dt * ( v**2 / rcoo + Sr + bvec_r )
    vec_a = v + dt * ( - u*v / rcoo + Sa + bvec_a )
    
    u1 = spar.linalg.spsolve( op_mat, vec_r )
    v1 = spar.linalg.spsolve( op_mat, vec_a )
    
    return u1, v1
    

Lap_ext = np.ones(  ( (Nr-1)*(Na-1) + 1, )*2  )
Lap_ext[:-1, :-1] = Lap_ins.toarray()
Lap_ext[-1,-1] = 0

Lap_sp = spar.csr_matrix( Lap_ext )

def Chorin( u, v ):
    div = u/rcoo + Dr @ u + ( Da @ v ) / rcoo
    source_ext = np.append( div, 0 )
    p = spar.linalg.spsolve( Lap_sp, source_ext )[:-1]
    
    u1 = u - Dr_ins @ p
    v1 = v - rinv_sp @ Da @ p
    
    return u1, v1



# -- Main Fluid Stepper

def fluid_stepper2d( u, v, ub, vb, Sr, Sa, mu, rho ):
    u0, v0 = fluid_momentum_step( u, v, ub, vb, Sr, Sa, mu, rho )
    return Chorin( u0, v0 )





# -----------------------------------------------------------------------------
# 


def unflatten( f, f0 ):
    mat = np.zeros_like(X)
    mat[1:-1, 1:-1] = f.reshape( np.shape(X[1:-1, 1:-1]) )
    
    avg = 0.5 * ( mat[:,1] + mat[:,-2] )
    mat[:,0] = mat[:,-1] = avg
    
    mat[0], mat[-1] = f0[0], f0[-1]
    
    return mat

def unflatten_ins( f ):
    mat = np.zeros_like(X)
    mat[1:-1, 1:-1] = f.reshape( np.shape(X[1:-1, 1:-1]) )
    
    mat[0], mat[-1] = mat[1], mat[-2]
    
    avg = 0.5 * ( mat[:,1] + mat[:,-2] )
    mat[:,0] = mat[:,-1] = avg
    
    return mat



# Convert VF from Polar form to rectangular form
def vec_translate( U, V ):
    fx = U * np.cos( A ) - V * np.sin( A )
    fy = U * np.sin( A ) + V * np.cos( A )
    return fx, fy



def boundary_dir( f0 ):
    mat = []
    for i in range(4):
        mat.append( np.zeros_like( X[1:-1, 1:-1] ) )
    # -- Radial boundaries
    mat[0][0], mat[2][0] = -0.5/dr * f0[0, 1:-1], 1/dr**2 * f0[0, 1:-1]
    mat[1][-1], mat[3][-1] = 0.5/dr * f0[-1, 1:-1], 1/dr**2 * f0[-1, 1:-1]
    
    return [ mat[i].flatten() for i in range(4) ]
    

# -- Explanation of output...
# -- 0: d/dr boundary term at r = r_inn (inner radial boundary)
# -- 1: d/dr boundary term at r = r_out (outer radial boundary)
# -- 2: d^2/dr^2 boundary term at r = r_inn
# -- 3: d^2/dr^2 boundary term at r = r_out



# -----------------------------------------------------------------------------
# Initial data

# u0 = 0.25*(  1 - np.cos( 2*np.pi * ( R - r_inn ) / Lr )  ) * ( 1 - np.cos(A) )

u0, v0 = np.zeros_like(R), np.zeros_like(R)

u,v = u0[1:-1, 1:-1].flatten(), v0[1:-1, 1:-1].flatten()


# -- Boundary terms

u_bnd = boundary_dir(u0)
v_bnd = boundary_dir(v0)


# Sources
Sr = np.zeros_like(u)

env_rad = 0.5*( 1 - np.cos( 2*np.pi * (R-r_inn)/Lr ) )

Sa0 = np.sin(2*A) * np.heaviside( np.pi/2 - A, 0 ) * env_rad
Sa = 0.5 * Sa0[1:-1, 1:-1].flatten()


# -----------------------------------------------------------------------------
# Plot Setup

fig, ax = plt.subplots( figsize=[10,10], dpi=60 )

fig.suptitle( 'Plot of Fluid Speed (or sqrt of Kinetic Energy)' )

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_aspect('equal')

grid0 = np.zeros_like(X)

hplot = ax.pcolormesh( X, Y, np.zeros_like(X), cmap='jet', vmin=0, vmax=1 )

fig.colorbar( hplot, ax=ax )

# qplot = ax.quiver( X[skip], Y[skip], grid0[skip], grid0[skip], pivot='middle' )

plt.show()


# -----------------------------------------------------------------------------
# Simulation


writer = PillowWriter( fps=10, metadata=None )


max_count = 0.1/dt
count = max_count


with writer.saving( fig, 'Radial Flow Plot.gif', 60 ):
    for t in tvals:
        
        if count+1 < max_count:
            count += 1
        else:
            count = 0
            
            hplot.remove()
            # qplot.remove()
            
            u_mat, v_mat = unflatten(u, u0), unflatten(v, v0)
            fx, fy = vec_translate( u_mat, v_mat )
            
            speed = np.sqrt( fx**2 + fy**2 )
            
            hplot = ax.pcolormesh( X, Y, speed, cmap='jet', vmin=0, vmax=1 )
            # qplot = ax.quiver( X[skip], Y[skip], fx[skip], fy[skip], pivot='middle' )
            
            writer.grab_frame()
        
        u,v = fluid_stepper2d( u, v, u_bnd, v_bnd, Sr, Sa, mu, rho )

writer.finish()








# for i in range(25):
#     u,v = fluid_stepper2d(u, v, Sr, Sa, mu, rho)


# u_mat, v_mat = unflatten(u), unflatten(v)

# fx, fy = vec_translate( u_mat, v_mat )

# qplot.remove()

# qplot = ax.quiver( X[skip], Y[skip], fx[skip], fy[skip], pivot='middle' )


