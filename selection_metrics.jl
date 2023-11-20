include("pso.jl")
include("ga.jl")
include("configs.jl")
using Evolutionary
using Metaheuristics
using Statistics
using DataFrames
using CSV

# --------------- GA --------------------

function metrics_selection(n)
    dfm = DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "PopSize (default)", "Minimum (avg)", "Iterations (avg)", "f(x) calls (avg)", "Selection", "Crossover (default)", "Mutation (default)"])

    for d in Configs.dimensions
        for s in Configs.selections
            configGA = GA.configsGA(Configs.rosenbrock,
            Configs.hypercube(d, Configs.minGA, Configs.maxGA),
            Configs.default_popsize, s, Configs.default_crossover, Configs.default_mutation,
            Configs.optsGA, Configs.metsGA)

            dfGA = GA.create_empty_dataframe()

            GA.add_collection!(dfGA, configGA, n)

            push!(dfm, [d, Configs.default_popsize, mean(dfGA[!, "Minimum"]), mean(dfGA[!, "Iterations"]), mean(dfGA[!, "f(x) calls"]), String(Symbol(s)), String(Symbol(Configs.default_crossover)), String(Symbol(Configs.default_mutation))])

            print("\n\n--------------------------------------------------------------------------------------\n\n")
            print(dfm)
        end
    end

    return dfm

end

n = 50
r = metrics_selection(n)
CSV.write("selection_results(n=" * String(Symbol(n)) * ")(time_limit=" * String(Symbol(Configs.time_limit)) * ").csv", r)