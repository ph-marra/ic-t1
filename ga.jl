module GA

import Pkg
Pkg.add("Evolutionary")
Pkg.add("DataFrames")

include("configs.jl")
using Evolutionary
using Random
using DataFrames
using Base

mutable struct configsGA
    f
    hyper
    populationSize::Int64
    selection
    crossover
    mutation
    opts
    mets
end

function x0_hypercube(hyper)
    return Evolutionary.BoxConstraints(min(hyper.a, hyper.b) .* ones(hyper.d), max(hyper.a, hyper.b) .* ones(hyper.d))
end

#rand(min(hyper.a, hyper.b) : max(hyper.a, hyper.b)) .* ones(hyper.d)
#(rand(Float64, hyper.d) .* abs(hyper.a - hyper.b)) .- min(hyper.a, hyper.b)

function create_empty_dataframe()
    return DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "PopSize", "Minimum", "Iterations", "f(x) calls", "Selection", "Crossover", "Mutation"])
end

function run_GA(config)
    return Evolutionary.optimize(
                        config.f,
                        x0_hypercube(config.hyper),
                        Evolutionary.GA(populationSize = config.populationSize, selection = config.selection,
                        crossover = config.crossover, mutation = config.mutation, metrics = config.mets),
                        config.opts)

end

function add_entry!(df, config, r)
    push!(df, [config.hyper.d, config.populationSize, Base.minimum(r), Evolutionary.iterations(r), Evolutionary.f_calls(r), config.selection, config.crossover, config.mutation])
end

function add_collection!(df, config, n)
    for i in range(1, n)
        r = run_GA(config)
        add_entry!(df, config, r)
    end
end

end