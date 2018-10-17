
function prune(foodweb::Matrix{Int64})

    S = size(foodweb, 1)

    # Identify basal species
    nprey = sum(foodweb, dims = 1)
    basal = findall(nprey[1, :] .== 0)

    # Identify basal species that have only specialized predators
    prune = zeros(0)
    for i in basal

        # Find all predators of basal species
        preds = findall(foodweb[i, :] .== 1)

        # If those predators each only have one prey, flag prey for removal
        if sum(nprey[preds]) == length(preds)
            append!(prune, i)
        end
    end

    # Prune any prey identified for pruning
    if length(prune) > 0
        retained = findall(x -> !(x in prune), 1:S)
        foodweb = foodweb[retained, retained]
    end

    return foodweb
end
