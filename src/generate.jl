
function niche_network(S::Int, C::Float64)

    cond = false
    adj = fill(0, S, S)
    n = fill(0.0, S)
    r = fill(0.0, S)
    c = fill(0.0, S)

    while !cond
        n = sort(rand(S))
        β = 1 / (2 * C) - 1
        r = n .* rand(Beta(1, β), S)

        # set r[position of min[η]] = 0, so that every web has at least one basal species
        r[1] = 0.0

        for pred in 1:S

            c[pred] = rand(Uniform(r[pred]/2, n[pred]))

            for prey in 1:S
                if ((c[pred] - r[pred] / 2) <= n[prey] <= (c[pred] + r[pred] / 2))
                    adj[prey, pred] = 1
                else
                    adj[prey, pred] = 0
                end
            end
        end

        cond = is_connected(DiGraph(adj))
    end

    return (fw = adj, n = n, r = r, c = c)
end

function prob_network(S, C)

    cond = false
    adj = fill(0, S, S)

    n = fill(0.0, S)
    r = fill(0.0, S)
    c = fill(0.0, S)

    while !cond

        n = sort(rand(S))
        β = 1 / (2 * C) - 1
        r = n .* rand(Beta(1, β), S)

        # set r[position of min[η]] = 0, so that every web has at least one basal species
        r[1] = 0.0

        @inbounds for pred in 1:S

            c[pred] = rand(Uniform(r[pred] / 2, n[pred]))

            for prey in 1:S
                ϕ = exp(-((n[prey] - c[pred]) / (r[pred] / 2)) ^ 2)
                adj[prey, pred] = rand(Bernoulli(ϕ), 1)[1]
            end
        end

        cond = is_connected(DiGraph(adj))
    end

    return (fw = adj, n = n, r = r, c = c)
end


function ϕ_mat(θ, m)

    ϕ = fill(0.0, m.S, m.S)

    @inbounds for pred in 1:m.nonbasal
        for prey in 1:m.S
            ϕ[prey, m.basal + pred] = 0.99 * exp(-((θ.n[prey] - θ.c[pred] * θ.n[m.basal + pred]) / (θ.r[pred] / 2)) ^ 2)
        end
    end

    return ϕ

end
