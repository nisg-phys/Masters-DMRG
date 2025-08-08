#!/usr/bin/env python
#
# Simple DMRG tutorial.  This code integrates the following concepts:
#  - Infinite system algorithm
#  - Finite system algorithm
#
# Copyright 2013 James R. Garrison and Ryan V. Mishmash.
# Open source under the MIT license.  Source code at
# <https://github.com/simple-dmrg/simple-dmrg/>

# This code will run under any version of Python >= 2.6.  The following line
# provides consistency between python2 and python3.
from __future__ import print_function, division  # requires Python >= 2.6

# numpy and scipy imports
import numpy as np
import scipy.linalg
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh,eigs  # Lanczos routine from ARPACK
from numpy import linalg as LA
import matplotlib.pyplot as plt
# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
from collections import namedtuple

Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:  #To check if the rows and columns are equal to the dimension of block
            return False
    return True

# This function should test the same exact things, so there is no need to
# repeat its definition.
is_valid_enlarged_block = is_valid_block

# Model-specific code for calculation two point correlation function of Transverse Ising Model
model_d = 2  # single-site basis size
neum_entropy_block=[]
reyne_entropy_block=[]
entropy=[]
SX_expectation=[]
SZ_expectation=[]
SX_i_expectation=[]
SZ_i_expectation=[]
energy_chain=[]
g=np.arange(0,3,0.1)
for i in g :
    Sz1 = np.array([[1, 0], [0, -1]], dtype='d')  # single-site S^z
    Sx1 = np.array([[0, 1], [1, 0]], dtype='d')  # single-site S^+

    H1 = - i*np.array([[0, 1], [1, 0]], dtype='d') -0.00000000001* np.array([[1, 0], [0, -1]], dtype='d')  # single-site H with small bias.

    def H2(Sz1,Sz2):  # two-site part of H
        """Given the operators S^z and S^+ on two sites in different Hilbert spaces
        (e.g. two blocks), returns a Kronecker product representing the
        corresponding two-site term in the Hamiltonian that joins the two sites.
        """
        Jz=1
        return (
            -Jz * kron(Sz1, Sz2)
        )

# conn refers to the connection operator, that is, the operator on the edge of
# the block, on the interior of the chain.  We need to be able to represent S^z
# and S^+ on that site in the current basis in order to grow the chain.
    initial_block = Block(length=1, basis_size=model_d, operator_dict={
        "H": H1,
        "conn_Sz": Sz1,
        "Sx_expectation": Sx1,
        "Sz_expectation": Sz1
    })

    def sys_enlarge_block(block):
        """This function enlarges the provided Block by a single site, returning an
        EnlargedBlock.
        """
        mblock = block.basis_size
        o = block.operator_dict

    # Create the new operators for the enlarged block.  Our basis becomes a
    # Kronecker product of the Block basis and the single-site basis.  NOTE:
    # `kron` uses the tensor product convention making blocks of the second
    # array scaled by the first.  As such, we adopt this convention for
    # Kronecker products throughout the code.
        enlarged_operator_dict = {
            "H": kron(o["H"], identity(model_d)) + kron(identity(mblock), H1) + H2(o["conn_Sz"],Sz1),
            "conn_Sz": kron(identity(mblock), Sz1),
            "Sx_expectation": kron(o["Sx_expectation"], identity(model_d)) + kron(identity(mblock), Sx1),
            "Sz_expectation": kron(o["Sz_expectation"], identity(model_d)) + kron(identity(mblock), Sz1)
        }

        return EnlargedBlock(length=(block.length + 1),
                             basis_size=(block.basis_size * model_d),
                             operator_dict=enlarged_operator_dict)
    def sys_enlarge_block3(block):      
        mblock=block.basis_size
        o=block.operator_dict
        enlarged_operator_dict = {
            "H": kron(o["H"], identity(model_d)) + kron(identity(mblock), H1) + H2(o["conn_Sz"],Sz1),
            "conn_Sz": kron(identity(mblock), Sz1),
            "Sx_expectation": kron(o["Sx_expectation"], identity(model_d)) + kron(identity(mblock), Sx1),
            "Sz_expectation": kron(o["Sz_expectation"], identity(model_d)) + kron(identity(mblock), Sz1),
            "Sx_i": kron(o["Sx_i"],identity(model_d)),
            "Sz_i": kron(o["Sz_i"],identity(model_d)),
        }
        
        return EnlargedBlock(length=(block.length + 1),
                             basis_size=(block.basis_size * model_d),
                             operator_dict=enlarged_operator_dict)

   

   

    def rotate_and_truncate(operator, transformation_matrix):
        """Transforms the operator to the new (possibly truncated) basis given by
        `transformation_matrix`.
        """
        return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

    def single_dmrg_step(sys, env, m):
        """Performs a single DMRG step using `sys` as the system and `env` as the
        environment, keeping a maximum of `m` states in the new basis.
        """
        assert is_valid_block(sys)
        assert is_valid_block(env)
        
             
        if sys.length>22:
             sys_enl= sys_enlarge_block3(sys) #calling function to add operator first time on a particular site

        else: 
             sys_enl = sys_enlarge_block(sys)                                  
        
        env_enl = sys_enlarge_block(env)
        assert is_valid_enlarged_block(sys_enl)
        assert is_valid_enlarged_block(env_enl)
        
        if sys.length==22:    #site minus one where the function is to be calculated.      
           sys_enl = sys_enlarge_block(sys)
           sys_enl.operator_dict["Sx_i"]=kron(identity(m),Sx1)
           sys_enl.operator_dict["Sz_i"]=kron(identity(m),Sz1)  
           assert is_valid_enlarged_block(sys_enl)
           assert is_valid_enlarged_block(env_enl)
        if sys.length==47:
           sys_enl = sys_enlarge_block3(sys)
           sys_enl.operator_dict["Sx_i"]=sys_enl.operator_dict["Sx_i"]*kron(identity(m),Sx1)
           sys_enl.operator_dict["Sz_i"]=sys_enl.operator_dict["Sz_i"]*kron(identity(m),Sz1)  
           assert is_valid_enlarged_block(sys_enl)
           assert is_valid_enlarged_block(env_enl)
        

    # Construct the full superblock Hamiltonian.
        m_sys_enl = sys_enl.basis_size
        m_env_enl = env_enl.basis_size
        sys_enl_op = sys_enl.operator_dict
        env_enl_op = env_enl.operator_dict
        superblock_hamiltonian = kron(sys_enl_op["H"], identity(m_env_enl)) + kron(identity(m_sys_enl), env_enl_op["H"]) + \
                             H2(sys_enl_op["conn_Sz"],env_enl_op["conn_Sz"])
        #print (superblock_hamiltonian)
        #print (superblock_hamiltonian.shape)

    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
        (energy,), psi0_n = eigsh(superblock_hamiltonian, k=1, which="SA")
        print(psi0_n.shape)

    # Construct the reduced density matrix of the system by tracing out the
    # environment
    #
    # We want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.  Since the environment (column) index updates most
    # quickly in our Kronecker product structure, psi0 is thus row-major ("C
    # style").
        #print ("shape of psi",psi0_n.shape)
        psi0 = psi0_n.reshape([sys_enl.basis_size, -1], order="C")
        rho_sys = np.dot(psi0, psi0.conjugate().transpose())
        rho_env = np.dot(psi0.conjugate().transpose(), psi0)
        s=rho_sys.shape
        print ("Shape of rho",s)
        h=2
    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.
        non_zero_evals=[]
        evals_sys, evecs_sys = np.linalg.eigh(rho_sys)
        evals_env, evecs_env = np.linalg.eigh(rho_env)
        for a in evals_sys :
            if a>=0:
               non_zero_evals.append(a)
        #entropy_block_sys= -np.trace(rho_sys*(scipy.linalg.logm(rho_sys)))
        #entropy_block_sys=entropy_block_sys.real
        entropy_block_sys2=-sum((non_zero_evals*np.log2(non_zero_evals)))
        #Reynee_entropy_block_sys=np.log(np.trace(LA.matrix_power(rho_sys,h)))/(1-h)
        possible_eigenstates_sys = []
        for eval_sys, evec_sys in zip(evals_sys, evecs_sys.transpose()):
            possible_eigenstates_sys.append((eval_sys, evec_sys))
        possible_eigenstates_sys.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
        my_m_sys = min(len(possible_eigenstates_sys), m)
        transformation_matrix_sys = np.zeros((sys_enl.basis_size, my_m_sys), dtype='d', order='F')
        for i, (eval_sys, evec_sys) in enumerate(possible_eigenstates_sys[:my_m_sys]):
            transformation_matrix_sys[:, i] = evec_sys

        truncation_error = 1 - sum([x[0] for x in possible_eigenstates_sys[:my_m_sys]])
        print("truncation error:", truncation_error)

        possible_eigenstates_env = []
        for eval_env, evec_env in zip(evals_env, evecs_env.transpose()):
            possible_eigenstates_env.append((eval_env, evec_env))
        possible_eigenstates_env.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
        my_m_env = min(len(possible_eigenstates_env), m)
        transformation_matrix_env = np.zeros((env_enl.basis_size, my_m_env), dtype='d', order='F')
        for i, (eval_env, evec_env) in enumerate(possible_eigenstates_env[:my_m_env]):
            transformation_matrix_env[:, i] = evec_env
 
        truncation_error = 1 - sum([x[0] for x in possible_eigenstates_env[:my_m_env]])
    


    # Rotate and truncate each operator.
        new_sys_operator_dict = {}
        for name, op in sys_enl.operator_dict.items():
            new_sys_operator_dict[name] = rotate_and_truncate(op, transformation_matrix_sys)
        new_env_operator_dict = {}
        for name, op in env_enl.operator_dict.items():
            new_env_operator_dict[name] = rotate_and_truncate(op, transformation_matrix_env)


        newsys_block = Block(length=sys_enl.length,
                         basis_size=my_m_sys,
                         operator_dict=new_sys_operator_dict) 
        #print ("Shape of Sx",newsys_block.operator_dict["Sx_expectation"].shape)
        newenv_block = Block(length=env_enl.length,
                         basis_size=my_m_env,
                         operator_dict=new_env_operator_dict)

        return newsys_block,newenv_block, energy,entropy_block_sys2,psi0_n
        

    def graphic(sys_block, env_block, sys_label="l"):
        """Returns a graphical representation of the DMRG step we are about to
        perform, using '=' to represent the system sites, '-' to represent the
        environment sites, and '**' to represent the two intermediate sites.
        """
        assert sys_label in ("l", "r")
        graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
        if sys_label == "r":
            # The system should be on the right and the environment should be on
           # the left, so reverse the graphic.
            graphic = graphic[::-1]
        return graphic

    def infinite_system_algorithm(L, m):
        sys_block = initial_block
        env_block= initial_block
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
        while 2 * sys_block.length < L:
            print("L =", sys_block.length * 2 + 2)
            sys_block,env_block, energy,entropy_block_sys2,psi0_n = single_dmrg_step(sys_block, env_block, m=m)
            print("E/L =", energy / (sys_block.length * 2))
            #print ("Entropy of the block of length =",sys_block.length, "using rho",entropy_block_sys.real)
            #print ("Reyni Entropy for system",Reynee_entropy_block_sys)
        SX = kron(sys_block.operator_dict["Sx_expectation"],identity(4*m))+kron(identity(4*m),env_block.operator_dict["Sx_expectation"])
        SX_expec=psi0_n.conjugate().transpose().dot(SX.dot(psi0_n))
        print (SX_expec)
        SX_i= kron(sys_block.operator_dict["Sx_i"],identity(4*m))
        print ("Sx_i",SX_i.shape)
        print ("psi0",psi0_n.shape)
        SX_iexpec=psi0_n.conjugate().transpose().dot(SX_i.dot(psi0_n))
        print ("Expectation value of order parameter Sx_i",SX_iexpec)
        #SZ = kron(sys_block.operator_dict["Sz_expectation"],identity(4*m))+kron(identity(4*m),env_block.operator_dict["Sz_expectation"])
        #SZ_expec=psi0_n.conjugate().transpose().dot(SZ.dot(psi0_n))
        #print ("Expectation value of order parameter Sz",SZ_expec)
    
    def finite_system_algorithm(L, m_warmup, m_sweep_list):
        assert L % 2 == 0  # require that L is an even number

    # To keep things simple, this dictionary is not actually saved to disk, but
    # we use it to represent persistent storage.
        block_disk = {}  # "disk" storage for Block objects

    # Use the infinite system algorithm to build up to desired size.  Each time
    # we construct a block, we save it for future reference as both a left
    # ("l") and right ("r") block, as the infinite system algorithm assumes the
    # environment is a mirror image of the system.
        block_sys = initial_block
        block_env = initial_block
        block_disk["l", block_sys.length] = block_sys
        block_disk["r", block_env.length] = block_env
        while 2 * block_sys.length < L:
        # Perform a single DMRG step and save the new Block to "disk"
            print(graphic(block_sys, block_env))
            #block_sys,block_env, energy,entropy_block_sys,Reynee_entropy_block_sys,psi0_n,entropy_block_sys2 = single_dmrg_step(block_sys, block_env, m=m_warmup)
            block_sys,block_env, energy,entropy_block_sys2,psi0_n = single_dmrg_step(block_sys, block_env, m=m_warmup)
            print("E/L =", energy / (block_sys.length * 2))
            block_disk["l", block_sys.length] = block_sys
            block_disk["r", block_env.length] = block_env

    # Now that the system is built up to its full size, we perform sweeps using
    # the finite system algorithm.  At first the left block will act as the
    # system, growing at the expense of the right block (the environment), but
    # once we come to the end of the chain these roles will be reversed.
        block_disk_sweep = {} 
        block_disk_sweep2 = {} 
        block_disk_sweep3 ={}
        global neum_entropy_block
        global reyne_entropy_block
        global entropy
        global SX_expectation
        global SZ_expectation
        global i
        global energy_chain
        global SX_i_expectation
        global SZ_i_expectation
        length_finite=[]
        #markers_finite={0.9:'ro',1.0:'bo',1.1:'yo'}
        #markers_finite={1.0:'go'}
        sys_label, env_label = "l", "r"
        sys_block = block_sys; del block_sys  # rename the variable
        for m in m_sweep_list:
            while True:
            # Load the appropriate environment block from "disk"
                env_block = block_disk[env_label, L - sys_block.length - 2]
                if env_block.length == 1:
                # We've come to the end of the chain, so we reverse course.
                    sys_block, env_block = env_block, sys_block
                    sys_label, env_label = env_label, sys_label

            # Perform a single DMRG step.
                print(graphic(sys_block, env_block, sys_label))
                #sys_block,env_block, energy,entropy_block_sys,Reynee_entropy_block_sys,psi0_n,entropy_block_sys2 = single_dmrg_step(sys_block, env_block, m=m)
                sys_block,env_block, energy,entropy_block_sys2,psi0_n = single_dmrg_step(sys_block,env_block, m=m)
                avg_energy= - energy / L
                print("E/L =", energy / L)
                #print ("Entropy of the block of length =",sys_block.length, "using rho",entropy_block_sys.real)
                #print ("Reyni Entropy for system",Reynee_entropy_block_sys)
               
    
            # Save the block from this step to disk.entropy_block_sys2
                block_disk[sys_label, sys_block.length] = sys_block
                #block_disk_sweep[sys_label, sys_block.length]=entropy_block_sys
                #block_disk_sweep2[sys_label,sys_block.length]=Reynee_entropy_block_sys
                block_disk_sweep3[sys_label,sys_block.length]=entropy_block_sys2 

            # Check whether we just completed a full sweep.
                if sys_label == "l" and 2 * sys_block.length == L:
                   break  # escape from the "while True" loopentropy_block_sys2
        
        #SX = kron(sys_block.operator_dict["Sx_expectation"],identity(4*m))+kron(identity(4*m),env_block.operator_dict["Sx_expectation"])
        #print ("shape of psi0",psi0_n.shape)
        #print ("shape of Sx",SX.shape)
        #SX_expec=psi0_n.conjugate().transpose().dot(SX.dot(psi0_n))
        #print (SX_expec)
        #print ("Expectation value of order parameter Sx",SX_expec)
        #SZ = kron(sys_block.operator_dict["Sz_expectation"],identity(4*m))+kron(identity(4*m),env_block.operator_dict["Sz_expectation"])
        #SZ_expec=psi0_n.conjugate().transpose().dot(SZ.dot(psi0_n))
        SX_i= kron(sys_block.operator_dict["Sx_i"],identity(4*m))
        SX_iexpec=psi0_n.conjugate().transpose().dot(SX_i.dot(psi0_n))
        SZ_i= kron(sys_block.operator_dict["Sz_i"],identity(4*m))
        SZ_iexpec=psi0_n.conjugate().transpose().dot(SZ_i.dot(psi0_n))
        #print ("psi0",psi0_n.shape)
        #print ("Expectation value of order parameter Sz",SZ_expec)
        #neum_entropy_block.append(block_disk_sweep['r',9])
        #reyne_entropy_block.append(block_disk_sweep2['r',9])
        #for j in np.arange(2,101,1):
        #entropy.append(block_disk_sweep3['r',100])
        #plt.plot(np.arange(2,101,1),entropy, markers_finite[i],label='%f'%i)
        #plt.legend()
        #entropy=[]
        #SX_expectation.append(SX_expec[0])
        #SZ_expectation.append(SZ_expec[0])
        SX_i_expectation.append(SX_iexpec[0,0])
        SZ_i_expectation.append(SZ_iexpec[0,0])
    if __name__ == "__main__":
        np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

        #infinite_system_algorithm(L=30,m=20)
        finite_system_algorithm(L=100, m_warmup=10, m_sweep_list=[30])
print (SX_i_expectation)
print (SZ_i_expectation)
#print (neum_entropy_block)
#print (reyne_entropy_block)
#print (energy_chain)
#b=np.arange(2,28,1)
#print (SX_expectation)
#print (SZ_expectation)
#plt.plot(g,neum_entropy_block,'r',label="Von neumann Entropy")
#plt.plot(g,reyne_entropy_block,'go',label="Renyi Entropy")
#b=np.arange(1,101,1)
#S=1/(2*g)
plt.plot(g,SX_i_expectation,'bo',label="<SiSj>")
plt.plot(g,SZ_i_expectation,'ro',label="<SiSj>")
plt.xlabel('J/g')
plt.ylabel('two point Correlation function')
#plt.plot(g,entropy,'bo',label="Entanglement entropy")
#plt.figure()
#plt.xlabel('magnetic field')
#plt.ylabel('Order Parameter')
#plt.plot(S,SX_expectation,'rD',label="<Sx>")
#plt.plot(S,SZ_expectation,'bD',label="<Sz>")
plt.legend()
plt.show()
