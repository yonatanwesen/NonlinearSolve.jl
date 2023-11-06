module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

using DiffEqBase, LinearAlgebra, LinearSolve, SparseArrays, SparseDiffTools
using FastBroadcast: @.., True, False
import ArrayInterface: restructure
import ForwardDiff

import ADTypes: AbstractFiniteDifferencesMode
import ArrayInterface: undefmatrix, matrix_colors, parameterless_type, ismutable,fast_scalar_indexing
import ConcreteStructs: @concrete
import EnumX: @enumx
import ForwardDiff: Dual
import LinearSolve: ComposePreconditioner, InvPreconditioner, needs_concrete_A
import RecursiveArrayTools: ArrayPartition,
    AbstractVectorOfArray, recursivecopy!, recursivefill!
import Reexport: @reexport
import SciMLBase: AbstractNonlinearAlgorithm, NLStats, _unwrap_val, has_jac, isinplace
import StaticArraysCore: StaticArray, SVector, SArray, MArray
import UnPack: @unpack
import DiffEqBase:calculate_residuals!

@reexport using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve

const AbstractSparseADType = Union{ADTypes.AbstractSparseFiniteDifferences,
    ADTypes.AbstractSparseForwardMode, ADTypes.AbstractSparseReverseMode}

abstract type AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractNewtonAlgorithm{CJ, AD} <: AbstractNonlinearSolveAlgorithm end

abstract type AbstractNonlinearSolveCache{iip} end

isinplace(::AbstractNonlinearSolveCache{iip}) where {iip} = iip

function SciMLBase.__solve(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
    alg::AbstractNonlinearSolveAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

function not_terminated(cache::AbstractNonlinearSolveCache)
    return !cache.force_stop && cache.stats.nsteps < cache.maxiters
end
get_fu(cache::AbstractNonlinearSolveCache) = cache.fu1

function SciMLBase.solve!(cache::AbstractNonlinearSolveCache)
    while not_terminated(cache)
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    return SciMLBase.build_solution(cache.prob, cache.alg, cache.u, get_fu(cache);
        cache.retcode, cache.stats)
end

include("utils.jl")
include("extension_algs.jl")
include("linesearch.jl")
include("raphson.jl")
include("trustRegion.jl")
include("levenberg.jl")
include("gaussnewton.jl")
include("dfsane.jl")
#include("pt_controller.jl")
include("pseudotransient.jl")
include("jacobian.jl")
#include("pt_controller.jl")
include("ad.jl")
include("default.jl")

import PrecompileTools

@static if VERSION >= v"1.10"
    PrecompileTools.@compile_workload begin
        for T in (Float32, Float64)
            prob = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))

            precompile_algs = (NewtonRaphson(), TrustRegion(), LevenbergMarquardt())

            for alg in precompile_algs
                solve(prob, alg, abstol = T(1e-2))
            end

            prob = NonlinearProblem{true}((du, u, p) -> du[1] = u[1] * u[1] - p[1], T[0.1],
                T[2])
            for alg in precompile_algs
                solve(prob, alg, abstol = T(1e-2))
            end
        end
    end
end

export RadiusUpdateSchemes

export NewtonRaphson, TrustRegion, LevenbergMarquardt, DFSane, GaussNewton, PseudoTransient
export LeastSquaresOptimJL, FastLevenbergMarquardtJL
export RobustMultiNewton, FastShortcutNonlinearPolyalg

export LineSearch

end # module
