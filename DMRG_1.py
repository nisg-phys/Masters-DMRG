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
import math
# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
from collections import namedtuple

Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

# This function should test the same exact things, so there is no need to
# repeat its definition.
is_valid_enlarged_block = is_valid_block

# Model-specific code for the Heisenberg XXZ chain
model_d = 2  # single-site basis size
energy_chain=[]
length=[]
Sx_expectation=[]
Sz_expectation=[]
g=np.arange(0,1.8,0.2)
#g=[1.0]
for i in g :
    Sz1 = np.array([[1, 0], [0, 1]], dtype='d')  # single-site S^z
    Sx1 = np.array([[0, 1], [1, 0]], dtype='d')  # single-site S^+

    H1 = -np.array([[0, 1], [1, 0]], dtype='d')   # single-site portion of H is zero

    def H2(Sz1,Sz2):  # two-site part of H
        """Given the operators S^z and S^+ on two sites in different Hilbert spaces
        (e.g. two blocks), returns a Kronecker product representing the
        corresponding two-site term in the Hamiltonian that joins the two sites.
        """
        Jz=1
        return (
                -i* kron(Sz1, Sz2)
        )

# conn refers to the connection operator, that is, the operator on the edge of
# the block, on the interior of the chain.  We need to be able to represent S^z
# and S^+ on that site in the current basis in order to grow the chain.
    initial_block = Block(length=1, basis_size=model_d, operator_dict={
        "H": H1,
        "conn_Sz": Sz1
       
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
            "conn_Sz": kron(identity(mblock), Sz1)
           
        }

        return EnlargedBlock(length=(block.length + 1),
                             basis_size=(block.basis_size * model_d),
                             operator_dict=enlarged_operator_dict)

    def env_enlarge_block(block):
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
            "H": kron(identity(model_d),o["H"]) + kron(H1,identity(mblock),) + H2(Sz1,o["conn_Sz"]),
            "conn_Sz": kron(Sz1,identity(mblock))
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

    # Enlarge each block by a single site.
        sys_enl = sys_enlarge_block(sys)
        env_enl = env_enlarge_block(env)

        assert is_valid_enlarged_block(sys_enl)
        assert is_valid_enlarged_block(env_enl)

    # Construct the full superblock Hamiltonian.
        m_sys_enl = sys_enl.basis_size
        m_env_enl = env_enl.basis_size
        sys_enl_op = sys_enl.operator_dict
        env_enl_op = env_enl.operator_dict
        superblock_hamiltonian = kron(sys_enl_op["H"], identity(m_env_enl)) + kron(identity(m_sys_enl), env_enl_op["H"]) + H2(sys_enl_op["conn_Sz"],env_enl_op["conn_Sz"])
        #print (superblock_hamiltonian)
        #print (superblock_hamiltonian.shape)

    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
        energy, psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")

    # Construct the reduced density matrix of the system by tracing out the
    # environment
    #
    # We want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.  Since the environment (column) index updates most
    # quickly in our Kronecker product structure, psi0 is thus row-major ("C
    # style").
        #print ("shape of psi",psi0_n.shape)
       
        psi0_n = psi0.reshape([sys_enl.basis_size, -1], order="C")
        rho_sys = np.dot(psi0_n, psi0_n.conjugate().transpose())
        rho_env = np.dot(psi0_n.conjugate().transpose(), psi0_n)
        
        
    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.
        non_zero_evals=[]
        evals_sys, evecs_sys = np.linalg.eigh(rho_sys)
        evals_env, evecs_env = np.linalg.eigh(rho_env)
        #print ("the eigenvalues of evals_sys is ",evals_sys)
        for a in evals_sys :
            if a>=0:
               non_zero_evals.append(a)
        entropy_block=-sum((non_zero_evals*np.log2(non_zero_evals)))
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

        return newsys_block,newenv_block, energy,psi0_n,entropy_block
       

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
        global energy_chain
        global length
        
        for k in m :
            sys_block = initial_block
            env_block= initial_block
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
            while 2 * sys_block.length < L:
                print("L =", sys_block.length * 2 + 2)
                sys_block,env_block, energy,psi0_n,entropy_block = single_dmrg_step(sys_block, env_block, m=m)
                avg_energy= energy/(sys_block.length * 2)
                energy_chain.append(avg_energy)
                length.append(sys_block.length * 2)
                print("E/L =", avg_energy / (sys_block.length * 2))
           
           
    
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
            block_sys,block_env, energy,psi0_n, entropy_block = single_dmrg_step(block_sys, block_env, m=m_warmup)
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
        global i
        global energy_chain
        global Sx_expectation
        global Sz_expectation
        length_finite=[]
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
                sys_block,env_block, energy,psi0_n, entropy_block = single_dmrg_step(sys_block,env_block, m=m)
                avg_energy= energy / L
                print("E/L =", avg_energy)
             
               
    
            # Save the block from this step to disk.entropy_block_sys2
                block_disk[sys_label, sys_block.length] = sys_block
                block_disk_sweep[sys_label, sys_block.length*2]=avg_energy
                block_disk_sweep3[sys_label, sys_block.length*2]=block_entropy
               
            # Check whether we just completed a full sweep.
                if sys_label == "l" and 2 * sys_block.length == L:
                   break  # escape from the "while True" loopentropy_block_sys2
        S_z= kron(sys_block.operator_dict["Sz_expectation"],identity(4*m))+kron(identity(4*m),env_block.operator_dict["Sz_expectation"])
        S_x= kron(sys_block.operator_dict["Sx_expectation"],identity(4*m))+kron(identity(4*m),env_block.operator_dict["Sx_expectation"])
        Sz_expec=(psi0.conjugate().transpose().dot(S_z.dot(psi0)))/L
        Sx_expec=(psi0.conjugate().transpose().dot(S_x.dot(psi0)))/L
        print ("SZ_expec",Sz_expec[0,0])
        print ("SZ_stagg",Sx_expec[0,0])
        Sz_expectation.append(Sz_expec[0,0])
        Sx_expectation.append(Sx_expec[0,0])
        energy_chain.append(avg_energy)
       
        
    if __name__ == "__main__":
        np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

        #infinite_system_algorithm(L=50,m=[20])
        finite_system_algorithm(L=30, m_warmup=10, m_sweep_list=[30])
print (Sz_expectation)
print (Sx_expectation)
plt.xlabel('J/Jz')
plt.ylabel('order parameter')
plt.plot(g2,Sz_expectation,'ro')
plt.plot(g2,Sx_expectation,'bo')
plt.legend()
plt.show()
plt.figure()
print  (energy_chain)
print (length)
plt.xlabel('system size')
plt.ylabel('Energy per site')
plt.plot(g,energy_chain,'r0')
plt.legend()
plt.show()
