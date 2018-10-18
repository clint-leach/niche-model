
@with_kw struct model

    foodweb::Matrix{Int64}
    S::Int64
    basal::Int64
    nonbasal::Int64
    top::Int64
    nontop::Int64
    reorder::Vector{Int64}
end

function model(foodweb::Matrix{Int64})

    # Prune prey with only specialist prey
    foodweb = prune(foodweb)

    # Identify basal species
    nprey = sum(foodweb, dims = 1)[1, :]
    basal = findall(nprey .== 0)

    # Identify top predators
    npred = sum(foodweb, dims = 2)[:, 1]
    top = findall(npred .== 0)

    # Identify everyone else
    middle = findall(x -> !(x in basal) && !(x in top), 1:S)

    # Sort the species
    order = vcat(basal, middle, top)

    model(foodweb = foodweb[order, order],
          S = size(foodweb, 1),
          basal = length(basal),
          nonbasal = size(foodweb, 1) - length(basal),
          top = length(top),
          nontop = size(foodweb, 1) - length(top),
          reorder = order)

end

function computeLL(θ, m)

    loglik = eltype(θ[1])(0.0)

    @inbounds for pred in 1:m.nonbasal
        for prey in 1:m.nontop
            ϕ = 0.99 * exp(-((θ.n[prey] - θ.c[pred] * θ.n[m.basal + pred]) / (θ.r[pred] / 2)) ^ 2)
            loglik += logpdf(Bernoulli(ϕ), m.foodweb[prey, m.basal + pred])
        end
    end

    return loglik
end

function (m::model)(θ)

    logpost = computeLL(θ, m)

    logpost += logpdf.(Beta(2, 8), θ.r) |> sum

    isnan(logpost) && return -Inf

    # @show logpost.value
    # @show ForwardDiff.value.(θ.c)

    return logpost
end
