using Distributions
using Parameters
using LightGraphs
using TransformVariables
using LogDensityProblems
using DynamicHMC
using ForwardDiff
using BenchmarkTools

include("model.jl")
include("generate.jl")

S = 50

nicheweb = niche_network(S, 0.1)

m = model(foodweb = nicheweb.fw)

θ = (n = rand(Beta(1, 1), S),
     r = rand(Beta(1, 1), S),
     c = rand(Beta(1, 1), S))

m(θ)

toθ = as((n = as(Array, as𝕀, S),
          r = as(Array, as𝕀, S),
          c = as(Array, as𝕀, S)))

lp_t = TransformedLogDensity(toθ, m)
lp∇ = ForwardDiffLogDensity(lp_t, chunk = ForwardDiff.Chunk{S}())

inits = inverse(toθ, θ)
logdensity(LogDensityProblems.ValueGradient, lp∇, inits)

chain, smplr = NUTS_init_tune_mcmc(lp∇, 2000, max_depth = 10)

NUTS_statistics(chain)
