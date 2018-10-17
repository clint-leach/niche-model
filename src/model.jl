
@with_kw struct model

    foodweb::Matrix{Int64}
    S::Int64
    basal::Int64
    nonbasal::Int64
    reorder::Vector{Int64}
end

function model(foodweb::Matrix{Int64})

    # Prune prey with only specialist prey
    foodweb = prune(foodweb)

    # Sort the species by number of prey
    nprey = sum(foodweb, dims = 1)
    cols = sort([1:length(nprey);], by = i -> nprey[i])

    # Identify number of basal species
    basal = sum(nprey .== 0)

    model(foodweb = foodweb[cols, cols],
          S = size(foodweb, 1),
          basal = basal,
          nonbasal = size(foodweb, 1) - basal,
          reorder = cols)

end

function computeLL(θ, m)

    loglik = eltype(θ[1])(0.0)

    @inbounds for pred in 1:m.nonbasal
        for prey in 1:m.S
            ϕ = 0.99 * exp(-((θ.n[prey] - θ.c[pred]) / (θ.r[pred] / 2)) ^ 2)
            loglik += logpdf(Bernoulli(ϕ), m.foodweb[prey, m.basal + pred])
        end
    end

    return loglik
end

function (m::model)(θ)

    logpost = computeLL(θ, m)

    logpost += logpdf.(Beta(1, 5), θ.r) |> sum

    isnan(logpost) && return -Inf

    # @show logpost.value
    # @show ForwardDiff.value.(θ.c)

    return logpost
end
