# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq
import scipy.sparse as sparse  
import scipy.sparse.linalg as sparse_linalg
from past.builtins import xrange

class SQG:
    
    RHO_0 = 1000.0
    G0    = 9.81
    THERMAL_COEFF = 2.0e-4
    #========================================================================#
    # Constructor: SQG class
    #========================================================================#
    def __init__(self, z_grid,rho_profile,F0):
    
        self.z_grid = z_grid
        self.nZ     = z_grid.size
        self.F0     = F0
        self.rho_profile = rho_profile
        self.vertical_modes_initialised = False
        #Build the vertical differention matrix

        self.vert_diff_matrix_tridiag = self.__Build_Vertical_Diff_Matrix()
    
    def Solve_Total_Streamfunction(self,SLA,SST,delta_x,delta_y):
        
        if not self.vertical_modes_initialised:
            rossby_rad,barotropic_mode,baroclinic_mode = self.Vertical_Eigenmodes()
            self.rossby_rad = rossby_rad
            self.barotropic_mode = barotropic_mode
            self.baroclinic_mode = baroclinic_mode
            self.vertical_modes_initialised = True
        
        surf_strfun     = self.Solve_Surface_Streamfunction((self.G0*self.THERMAL_COEFF)*SST,delta_x,delta_y)
        interior_strfun = self.Solve_Interior_Streamfunction((self.G0/self.F0)*SLA,self.barotropic_mode,self.baroclinic_mode)
        total_strfun    = surf_strfun + interior_strfun
    
        return surf_strfun, interior_strfun, total_strfun
        
    def Vertical_Eigenmodes(self):
        '''
        PUBLIC: Vertical_Eigenmodes
        Solves the special case of the Sturm-Liouville problem:
        
            d/dz[S(z)dpdz] = -gamma^2 p 
        
        with 
        
            dpdz = 0 @ z=0,-H
        
        for a general density profile S(z) = 1/rho(z) using second order 
        finite differences. The precomputed vertical differentiation matrix
        is used to solve the eigenvalue problem. 
        ''' 
        delta_z = self.z_grid[0]-self.z_grid[1]
        #Put the tridiagonal matrix into dense matrix form to 
        #compute the eigenvalues and eigenvectors
        #D2z = sparse.diags(self.vert_diff_matrix_tridiag[0],offsets=-1) + \
        #      sparse.diags(self.vert_diff_matrix_tridiag[1],offsets=0)  + \
        #      sparse.diags(self.vert_diff_matrix_tridiag[2],offsets=1)
              
        D2z = np.diag(self.vert_diff_matrix_tridiag[0],k=-1) + \
              np.diag(self.vert_diff_matrix_tridiag[1],0)    + \
              np.diag(self.vert_diff_matrix_tridiag[2],k=1)
        #eigenvalues,eigenvectors = sparse_linalg.eigs(D2z,k=10,which='SM')
        
        eigenvalues,eigenvectors = linalg.eig(D2z)
        sort_idx = np.argsort(np.sqrt(-eigenvalues))
        #We get only the first two eigenmodes, as these correspond to the 
        #
        rossby_def_wavenumbers = -eigenvalues[sort_idx[0:2]] #[sort_idx[0:2]]
        vertical_modes         =  eigenvectors[:,sort_idx[0:2]] # [:,sort_idx[0:2]]
        del eigenvectors,eigenvalues
        
        #Normalise the eigenmodes such that the <v_m,v_n> = d_mn
        #where <:,:> is the inner product at d_mn is the delta function 
        delta_z_norm = 1.0*delta_z/np.abs(self.z_grid[-1])
        
        for i_mode in range(0,2):
            alpha= np.dot( (vertical_modes[:,i_mode]*delta_z_norm).T , vertical_modes[:,i_mode] )
            vertical_modes[:,i_mode]=vertical_modes[:,i_mode]/np.sqrt(alpha);
            if vertical_modes[1,i_mode]<0:
                vertical_modes[:,i_mode] = -vertical_modes[:,i_mode]
        #return the barotropic and baroclinic models and their rossby radii
        return 1.0/np.sqrt(rossby_def_wavenumbers),vertical_modes[:,0],vertical_modes[:,1]
 
    def Solve_Surface_Streamfunction(self,surf_buoyancy,delta_x,delta_y,solve_interior=True):
        
        '''
        PUBLIC: Solve_Surface_Streamfunction 
        Inverts the homogenous quasi-geostrophic potential vorticity equation
        \/^2p + d/dz(f_0^2/N^2 dp/dz) = 0 
        in 3-dimensions using mixed spectral/finite difference method
        '''        
        nY,nX = surf_buoyancy.shape
        #Enforce periodicity of the input surface field by mirror symetry
        mirror_surf_buoyancy = self.__Mirror_Field(surf_buoyancy)
        
        #Get the 2D Fourier Transform of the input field in the zonal and 
        #meridional directions (horizontal) and the associated wavenumbers 
        #for the spectral part of the solution
        surf_buoyancy_FT = fft2(mirror_surf_buoyancy)  
        kx = fftfreq(2*nX,d=delta_x)  *2.0*np.pi       #Zonal wavenumber
        ly = fftfreq(2*nY,d=delta_y)  *2.0*np.pi       #Meridional wavenumber
        KX,LY = np.meshgrid(kx, ly)                    #Put them on a grid
        K2 = KX*KX + LY*LY                             #Wavenumber magnitude
        K2[0,0] = 1.0

        S   = self.F0*self.F0*self.RHO_0/self.G0 #stratification parameter 
        delta_rho_surf = self.rho_profile[1] - self.rho_profile[0]
        RHS = np.zeros(self.nZ,dtype='complex128') 
        SQG_streamfunction_FFT = np.zeros([self.nZ,2*nY,2*nX],dtype=surf_buoyancy_FT.dtype)
        #Main loop: loop over all wavenumbers in the domain and solve the 
        #vertical part of the equation
        #print 'Solving in the vertical direction'
        for iY in range(0,2*nY):
           # print 'iY= ', iY, ' of ', 2*nY 
            for iX in range(0,2*nX):
                
                #The horizontal laplace operator in wavenumber space is simply
                #the square of the local wavenumeber
                DK = -K2[iY,iX]*np.ones(self.nZ,dtype='complex128')
                #DK[0]  = 0.0
                #DK[-1] = 0.0
                #=======================================#
                #Application of the Neumann conditions
                #at the surface: dp/dz = b_surface/f0                
                #=======================================#
                top_bc = S*surf_buoyancy_FT[iY,iX]/(self.F0*delta_rho_surf)
                #top_bc = surf_buoyancy_FT[iY,iX]/self.F0

                RHS_current = RHS.copy() + 0.0j
                RHS_current[0]    = 0.5*RHS_current[0] -  top_bc 
                RHS_current[self.nZ-1] = 0.0
                #self.vert_diff_matrix_tridiag[1][0] =  1.0/(self.z_grid[0]-self.z_grid[1])
                #self.vert_diff_matrix_tridiag[2][0] = -1.0/(self.z_grid[0]-self.z_grid[1])
                #
                #self.vert_diff_matrix_tridiag[0][0] =  1.0/(self.z_grid[0]-self.z_grid[1])
                #self.vert_diff_matrix_tridiag[1][0] = -1.0/(self.z_grid[0]-self.z_grid[1])
                

                #Diff_Matrix = sparse.diags(self.vert_diff_matrix_tridiag[0],offsets=-1)     + \
                #              sparse.diags(self.vert_diff_matrix_tridiag[1] + DK,offsets=0) + \
                #              sparse.diags(self.vert_diff_matrix_tridiag[2],offsets=1) + 0.0j
                #Solve the linear system to get the resulting surface 
                #streamfunction (in wavenumber space)
                SQG_streamfunction_FFT[:,iY,iX] = self.__TriDiagonalSolver(
                                                  self.vert_diff_matrix_tridiag[0], 
                                                  self.vert_diff_matrix_tridiag[1] + DK, 
                                                  self.vert_diff_matrix_tridiag[2],
                                                  RHS_current)
                #SQG_streamfunction_FFT[:,iY,iX]  = sparse_linalg.spsolve(Diff_Matrix, RHS_current)

        #print 'Inverse Fourier Transform to real space'
        SQG_streamfunction = np.zeros_like(SQG_streamfunction_FFT)
        for iZ in range(0,self.nZ):
            SQG_streamfunction[iZ,:,:] = ifft2(SQG_streamfunction_FFT[iZ,:,:])
        #Finished inverting the QGPV equation for the homogenous problem.  
        if solve_interior:
            self.SQG_streamfunction_FFT = SQG_streamfunction_FFT
        #Return the trimmed version
        return SQG_streamfunction[:,0:nY,0:nX].real
    
    def Solve_Interior_Streamfunction(self,surf_dyn_height,barotropic_mode,baroclinic_mode):
        
        '''
        PUBLIC: Solve_Interior_Streamfunction 
        Approximates the interior streamfunction as the weighted sum of the
        barotropi and 1st baroclinic modes (which are orthogonal, but do not
        form a complete basis set that spans to the vertical space, as we do
        not have enough information to constrain the higher modes).
        
        interior_streamfunction(z) = sum_n (A_n * F_n(z)) = A_0 * F_0(z) +A_1 * F_1(z) 
        where: F_n are the nth eigenfunction (n=0 -> barotropic, n=1 -> 1st baroclinic)
              A_m is the expansion coefficients.

        To determine the weights, we note that, at the surface, the sum of the
        interior and surface streamfunctions must equal the surface dynamic 
        height. At the bottom boundary, the sum of the interior and surface 
        streamfunctions should sum to zero. Thus, we determine the weight 
        coefficients by solving the linear system:
        
        A_0 F_0(z=-H) +  A_1 F_1(z=-H) = dyn_height - surface_streamfunction(z=0)
        A_0 F_0(z=-H) +  A_1 F_1(z=-H) = - surface_streamfunction(z=-H)
        
        which has an analytic solution.  
        '''        
        nY,nX = surf_dyn_height.shape
        #Enforce periodicity of the input surface field by mirror symetry
        mirror_dyn_height = self.__Mirror_Field(surf_dyn_height)
        
        #Get the 2D Fourier Transform of the surface field 
        FT_dyn_height = fft2(mirror_dyn_height)
        FT_interior_streamfunction = np.zeros_like(self.SQG_streamfunction_FFT)
        
        #Build the maxtrix
        #LHS_system = np.zeros([2,2],dtype='complex64')
        #LHS_system[0,0] =barotropic_mode[0]+0.0j
        #LHS_system[0,1] =baroclinic_mode[0]+0.0j
        #
        #LHS_system[1,0] =barotropic_mode[self.nZ-1]+0.0j
        #LHS_system[1,1] =baroclinic_mode[self.nZ-1]+0.0j
        #
        #RHS = np.zeros([2],dtype='complex64')
        
        for iY in range(0,2*nY):
            for iX in range(0,2*nX):
         #       RHS[0] = FT_dyn_height[iY,iX]-self.SQG_streamfunction_FFT[0,iY,iX]
         #       RHS[1] = -self.SQG_streamfunction_FFT[self.nZ-1,iY,iX]
          #      coeffs = linalg.solve(LHS_system, RHS)
                
                #Solve the 2x2 linear system analytically for every wavenumber
                #to determine the expansion coefficients
                coeff0,coeff1 = self.__Solve_2x2_Linear(
                                    barotropic_mode[0]+0.0j,barotropic_mode[self.nZ-1]+0.0j,
                                    baroclinic_mode[0]+0.0j,baroclinic_mode[self.nZ-1]+0.0j,
                                    FT_dyn_height[iY,iX]-self.SQG_streamfunction_FFT[0,iY,iX],-self.SQG_streamfunction_FFT[self.nZ-1,iY,iX]+0.0j)
                #Reconstruct the interior streamfunction in wavenumber space
                FT_interior_streamfunction[:,iY,iX] =coeff0 *barotropic_mode + coeff1*baroclinic_mode
                #FT_interior_streamfunction[:,iY,iX] =coeffs[0] * barotropic_mode + coeffs[1] * baroclinic_mode
        
        #Inverse Fourier Transform to convert back to the physical domain
        interior_streamfunction = np.zeros_like(FT_interior_streamfunction)
        for iZ in range(0,self.nZ):
            interior_streamfunction[iZ,:,:] = ifft2(FT_interior_streamfunction[iZ,:,:])
            
        #Finished the projection onto the barotropic and baroclinic modes.
        #We return the trimmed function 
        return  interior_streamfunction[:,0:nY,0:nX].real
    
    def Modify_Density(self,rho_profile):
        
        '''
        PUBLIC: Modify_Density
        
        Function modifies the density profile used as the background state in
        the SQG formulation. 
        '''    
        #Replace the old density profile with the new 
        self.rho_profile = rho_profile
        
        #Rebuild the differentiation matricies and compute the eigenvalues/
        #eigenvectors
        self.vert_diff_matrix_tridiag = self.__Build_Vertical_Diff_Matrix()

        rossby_rad,barotropic_mode,baroclinic_mode = self.Vertical_Eigenmodes()
        self.rossby_rad = rossby_rad
        self.barotropic_mode = barotropic_mode
        self.baroclinic_mode = baroclinic_mode
        self.vertical_modes_initialised = True

    #========================================================================#
    # PRIVATE FUNCTIONS
    #  - Build_Vertical_Diff_Matrix: used to create the tridiagonal matrix 
    #                                  that is the discreet version of the 
    #                                  vortex streching operator.
    #  
    #  - Mirror_Field: Mirror a two dimensional field in 4 quadrants to 
    #                  force periodicity 
    #========================================================================#
    def __Build_Vertical_Diff_Matrix(self,top_bcs='N',bottom_bcs='N'):    
        '''
        PRIVATE: Build Vertical_Diff_Matrix
        Function builds the tridiagonal differentiation matrix for the vortex 
        strectching term in the quasi-geostrophic equations using a 2nd order 
        finite difference scheme on a staggered grid. The implementation follows
        Smith (2007) and operates directly on the density profile (as opposed to 
        the buoyancy frequency) to avoid problems with the differentiation of 
        real oceanographic data, which can be noisy.
         
        The stretching operator is:
            d/dz (f0^2/N^2 d/dz )
        Since the square of the bouyancy freqency, N^2, is given by: 
            N^2 = (g0/rho_0) * d(rho)/dz
        then:
            1/N^2d/dz = rho_0/g0 [ 1/ (d(rho)/dz) * d/dz] = rho_0/g0 d/d(rho)
            
        '''
        
        #Vertical grid spacing. Z should be negative downwards (ie. z=-depth)
        delta_z = self.z_grid[0]-self.z_grid[1]
        
        delta_rho = self.rho_profile[1:self.nZ] - self.rho_profile[0:self.nZ-1]
        S         = self.F0*self.F0*self.RHO_0/self.G0
    
    
        upper_diag = S/(delta_z*delta_rho) + 0.0j
        lower_diag = S/(delta_z*delta_rho) + 0.0j
        main_diag   = np.zeros([self.nZ],dtype='complex128')
        main_diag[1:self.nZ-1]  = -(S/delta_z) * ( (1.0/delta_rho[0:delta_rho.size-1]) + (1.0/delta_rho[1:delta_rho.size]) )
    
    
        if top_bcs.upper()=='N':
            main_diag[0]  =  -S/(delta_z * delta_rho[0]) + 0.0j
            upper_diag[0] =   S/(delta_z * delta_rho[0]) + 0.0j
        elif top_bcs.upper()=='D':
            main_diag[0]  =  1.0 + 0.0j
            upper_diag[0] =  0.0 + 0.0j

        if bottom_bcs.upper()=='N':
            main_diag[self.nZ-1]  =  -S/(delta_z * delta_rho[delta_rho.size-1]) + 0.0j
            lower_diag[self.nZ-2] =   S/(delta_z * delta_rho[delta_rho.size-1]) + 0.0j
        elif bottom_bcs.upper()=='D':
            main_diag[0]  =  1.0 + 0.0j
            upper_diag[0] =  0.0 + 0.0j
    
        return lower_diag, main_diag,upper_diag


    def __Mirror_Field(self,input_field):
        '''
        PRIVATE: __Mirror_Field
        Function mirrors a field in 4 quadrants in order enforce periodicity 
        for 2 dimensional FFTs. This is the 2D ananlogue of periodic extension
        of a 1D time series and consists of mapping a nY x nX field to a 
        2nY x 2nX field by flipping along the axes as indicated below
        
        -----------------------
        |          |          |
        |     o    |     o    |
        |     3    |     4    |
        |     ^    |     ^    |
        ----- |----------|------
        |          |          |
        |    x     |     o    |
        |    1     ->    2    |
        |          |          |
        -----------------------
        x = original field
        o = flipped 
        '''
        #Size of the input grid
        nY,nX=input_field.shape 
        mirror_field = np.zeros([2*nY,2*nX],dtype=input_field.dtype)
        
        
        mirror_field[0:nY,0:nX]       = input_field            #1st quadrant
        mirror_field[0:nY,nX:2*nX]    = input_field[:,::-1]    #2nd quadrant
        mirror_field[nY:2*nY,0:nX]    = input_field[::-1,:]    #3rd quadrant
        mirror_field[nY:2*nY,nX:2*nX] = input_field[::-1,::-1] #4th quadrant
        
        return mirror_field
    
    def __Solve_2x2_Linear(self,a1,a2,b1,b2,RHS_1,RHS_2):
        '''
        Simple, Cramer's Rule like solution of a 2x2 linear system 
        |a1 b1| (x1) = RHS_1
        |a2 b2| (x2) = RHS_2
        '''
        x1 = (b2*RHS_1 - b1*RHS_2) / (b2*a1-b1*a2)
        x2 = (a1*RHS_2 - a2*RHS_1) / (b2*a1-b1*a2)

        return x1,x2


    ## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
    def __TriDiagonalSolver(self,lower_diag, main_diag, upper_diag, RHS):
        '''
        PRIVATE: TriDiagonalSolver, input arguments can be NumPy array type or Python list type.
        Solves the expression Ax=b, where A is an NxN tridiagonal matrix
        refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
        '''
        nf = len(RHS) # number of equations
        ac, bc, cc, dc = map(np.array, (lower_diag, main_diag, upper_diag, RHS)) # copy arrays
        for it in xrange(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]
        	    
        xc = bc
        xc[-1] = dc[-1]/bc[-1]

        for il in xrange(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

        return xc