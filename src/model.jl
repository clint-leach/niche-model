
@with_kw struct model

    foodweb::Matrix{Int64}
    S::Int64 = size(foodweb)[1]

end


function computeLL(θ, m)

    loglik = eltype(θ[1])(0.0)

    @inbounds for pred in 1:m.S
        for prey in 1:m.S
            ϕ = exp(-((θ.n[prey] - θ.c[pred]) / (θ.r[pred] / 2)) ^ 2)
            loglik += logpdf(Bernoulli(ϕ), m.foodweb[prey, pred])
        end
    end

    return loglik
end

function (m::model)(θ)

    logpost = computeLL(θ, m)

    logpost == -Inf && return -Inf

    logpost += logpdf.(Beta(1, 1), θ.n) |> sum
    logpost += logpdf.(Beta(1, 1), θ.r) |> sum
    logpost += logpdf.(Beta(1, 1), θ.c) |> sum

end
