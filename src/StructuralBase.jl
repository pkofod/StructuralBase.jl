module StructuralBase

using Optim, MDPTools, Reexport
@reexport using StatsBase
const AbstractUtility = MDPTools.AbstractUtility
const AbstractState = MDPTools.AbstractState
export EstimationMethod, EstimationResults, AbstractUtility,
        AbstractTrace, ConvergenceInfo, LinearUtility,
        AbstractState, hessian, gradient, loglikelihood

import StatsBase: loglikelihood
import Base: gradient


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
    ∇loglikelihood::Vector{Tf}
    ∇²loglikelihood::Matrix{Tf}
    tdf::Optim.TwiceDifferentiableFunction #TwiceDifferentiableFunction
    coef::Vector{Tf}
    conv::ConvergenceInfo
    trace::Any#::Optim.MultivariateOptimizationResults FIXME should be MultivariateOptimizationResults
    nobs
    meta
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

nobs(res) = res.nobs
loglikelihood(res) = res.loglikelihood
loglikelihood(res, x) = -res.tdf.f(x)*nobs(res)
gradient(res) = res.∇loglikelihood
function gradient(res, x)
    g = similar(x)
    res.tdf.g!(x, g)
    -g*nobs(res)
end
hessian(res::EstimationResults) = res.∇²loglikelihood
function hessian(res::EstimationResults, x)
    n = length(x)
    h = zeros(n, n)
    res.tdf.h!(x, h)
    -h*nobs(res)
end
coef(res::EstimationResults) = res.coef
coef(res, i) = coef(res)[i]
vcov(res::EstimationResults) = inv(-hessian(res))
stderr(res::EstimationResults) = sqrt(diag(vcov(res)))
stderr(res::EstimationResults, i) = sqrt(diag(vcov(res)))[i]
tstat(res, i, x) = (coef(res, i)-x)/(stderr(res, i))
tstat(res, i) = tstat(res, i, 0)
tstat(res, x::Vector) = [tstat(res, i, x[i]) for i = 1:length(x)]
tstats(res) = tstat(res, zeros(coef(res)))
end # module
