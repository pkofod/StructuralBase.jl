module StructuralBase

using Optim, MDPTools, Reexport
@reexport using StatsBase
const AbstractUtility = MDPTools.AbstractUtility
const AbstractState = MDPTools.AbstractState
export EstimationMethod, EstimationResults, AbstractUtility,
        AbstractTrace, ConvergenceInfo, LinearUtility,
        AbstractState

import StatsBase: loglikelihood


abstract EstimationMethod
abstract ConvergenceInfo

# --- AbstractTrace ---

abstract AbstractTrace
abstract Trace


TraceNFXP(K, n_var) = TraceNFXP(zeros(K, n_var),
                                zeros(K),
                                zeros(K),
                                zeros(K),
                                zeros(K))

type EstimationResults{T<:EstimationMethod, Tf<:Real}
    E::T
    loglikelihood::Tf
    ∇²loglikelihood::Matrix{Tf}
    conv::ConvergenceInfo
    trace::Any#::Optim.MultivariateOptimizationResults FIXME should be MultivariateOptimizationResults
    std_err
end


type ConvergenceNPL <: ConvergenceInfo
    maxK::Int64
    # Outer loop
    flag::Bool
    outer::Int64
    norm_θ::Float64
    # Inner loop
    ll::Float64
    norm_grad::Float64
    iter_maxlike::Int64
end

ConvergenceNPL() = ConvergenceNPL(0, false, 0, 0., 0., 0., 0)


type TraceNPL <: Trace
    θ::Matrix{Float64}
    norm_Δθ::Vector{Float64}
    ll::Vector{Float64}
    Δll::Vector{Float64}
    norm_g::Vector{Float64}
end

TraceNPL(K, n) = TraceNPL(zeros(K, n),
                          zeros(K),
                          zeros(K),
                          zeros(K),
                          zeros(K))

loglikelihood(res) = res.loglikelihood
vcov(res) = res.vcov
end # module
