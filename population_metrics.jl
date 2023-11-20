include("pso.jl")
include("ga.jl")
include("configs.jl")
using Evolutionary
using Metaheuristics
using Statistics
using DataFrames
using CSV

# --------------- GA --------------------

function metrics_population(n)
    dfm = DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "PopSize", "Minimum (avg)", "Iterations (avg)", "f(x) calls (avg)", "Selection (default)", "Crossover (default)", "Mutation (default)"])

    for d in Configs.dimensions
        for p in Configs.popsizes
            configGA = GA.configsGA(Configs.rosenbrock,
            Configs.hypercube(d, Configs.minGA, Configs.maxGA),
            p, Configs.default_selection, Configs.default_crossover, Configs.default_mutation,
            Configs.optsGA, Configs.metsGA)

            dfGA = GA.create_empty_dataframe()

            GA.add_collection!(dfGA, configGA, n)

            push!(dfm, [d, p, mean(dfGA[!, "Minimum"]), mean(dfGA[!, "Iterations"]), mean(dfGA[!, "f(x) calls"]), String(Symbol(Configs.default_selection)), String(Symbol(Configs.default_crossover)), String(Symbol(Configs.default_mutation))])

            print("\n\n--------------------------------------------------------------------------------------\n\n")
            print(dfm)
        end
    end

    return dfm

end

n = 50
r = metrics_population(n)
CSV.write("population_results(n=" * String(Symbol(n)) * ")(time_limit=" * String(Symbol(Configs.time_limit)) * ").csv", r)