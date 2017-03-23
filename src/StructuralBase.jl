module StructuralBase

using Optim, MDPTools, Reexport
@reexport using StatsBase
const AbstractUtility = MDPTools.AbstractUtility
const AbstractState = MDPTools.AbstractState
export EstimationMethod, EstimationResults, AbstractUtility,
        AbstractTrace, ConvergenceInfo, LinearUtility,
        AbstractState, hessian, gradient, loglikelihood,
        tstat

import StatsBase: loglikelihood
import Base: gradient


abstract EstimationMethod
abstract ConvergenceInfo

# --- AbstractTrace ---

abstract AbstractTrace
abstract Trace


type EstimationResults{T<:EstimationMethod, Tf<:Real}
    E::T
    loglikelihood::Tf
    ∇loglikelihood::Vector{Tf}
    ∇²loglikelihood::Matrix{Tf}
    tdf::Optim.TwiceDifferentiable #TwiceDifferentiableFunction
    coef::Vector{Tf}
    conv::ConvergenceInfo
    trace::Any#::Optim.MultivariateOptimizationResults FIXME should be MultivariateOptimizationResults
    nobs
    meta
end

# computational details
convinfo(res::EstimationResults) = res.conv
outer_iterations(res::EstimationResults) = convinfo(res).outer
inner_iterations(res::EstimationResults) = inner_iterations(convinfo(res))

newton_iterations(res::EstimationResults) = newton_iterations(convinfo(res))
contraction_iterations(res::EstimationResults) = contraction_iterations(convinfo(res))

# loglikelihood
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

# coefficients and inference
coef(res::EstimationResults) = res.coef
coef(res, i) = coef(res)[i]
vcov(res::EstimationResults) = inv(-hessian(res))
stderr(res::EstimationResults) = sqrt.(diag(vcov(res)))
stderr(res::EstimationResults, i) = sqrt.(diag(vcov(res)))[i]
tstat(res, i, x) = (coef(res, i)-x)/(stderr(res, i))
tstat(res, i) = tstat(res, i, 0)
tstat(res, x::Vector) = [tstat(res, i, x[i]) for i = 1:length(x)]
tstats(res) = tstat(res, zeros(coef(res)))
end # module
