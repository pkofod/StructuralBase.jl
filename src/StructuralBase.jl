module StructuralBase

using Optim, MDPTools
const AbstractUtility = MDPTools.AbstractUtility
const AbstractState = MDPTools.AbstractState
export EstimationMethod, EstimationResults, AbstractUtility,
        AbstractTrace, ConvergenceInfo

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

type EstimationResults{T<:EstimationMethod}
    E::T
    conv::ConvergenceInfo
    trace::Optim.MultivariateOptimizationResults
end


type EstimationResultsNPL
    E::EstimationMethod
    conv::ConvergenceInfo
    trace
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

end # module
