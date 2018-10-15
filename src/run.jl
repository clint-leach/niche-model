using Distributions
using Parameters
using LightGraphs
using TransformVariables
using LogDensityProblems
using DynamicHMC
using ForwardDiff
using BenchmarkTools
using Plots

include("model.jl")
include("generate.jl")

S = 10

nicheweb = niche_network(S, 0.1)

heatmap(nicheweb.fw, legend = false)

m = model(nicheweb.fw)

heatmap(m.foodweb, legend = false)

θ = (n = nicheweb.n[m.reorder],
     r = nicheweb.r[m.reorder[(m.basal + 1):end]],
     c = nicheweb.c[m.reorder[(m.basal + 1):end]])

m(θ)

toθ = as((n = as(Array, as𝕀, m.S),
          r = as(Array, as𝕀, m.nonbasal),
          c = as(Array, as𝕀, m.nonbasal)))

lp_t = TransformedLogDensity(toθ, m)
lp∇ = ForwardDiffLogDensity(lp_t, chunk = ForwardDiff.Chunk{24}())

inits = inverse(toθ, θ)
logdensity(LogDensityProblems.ValueGradient, lp∇, inits)

chain, smplr = NUTS_init_tune_mcmc(lp∇, 500, max_depth = 10)

NUTS_statistics(chain)

posterior = transform.(Ref(lp∇.transformation), get_position.(chain))

n = mean(first, posterior)

foo = zip(posterior...) |> collect

scatter(θ.n, n)
