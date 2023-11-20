include("pso.jl")
include("ga.jl")
include("configs.jl")
using Evolutionary
using Metaheuristics
using Statistics
using DataFrames
using CSV

# --------------- GA --------------------

function metrics_mutation(n)
    dfm = DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "PopSize (default)", "Minimum (avg)", "Iterations (avg)", "f(x) calls (avg)", "Selection (default)", "Crossover (default)", "Mutation"])

    for d in Configs.dimensions
        for m in Configs.mutations
            configGA = GA.configsGA(Configs.rosenbrock,
            Configs.hypercube(d, Configs.minGA, Configs.maxGA),
            Configs.default_popsize, Configs.default_selection, Configs.default_crossover, m,
            Configs.optsGA, Configs.metsGA)

            dfGA = GA.create_empty_dataframe()

            GA.add_collection!(dfGA, configGA, n)

            push!(dfm, [d, Configs.default_popsize, mean(dfGA[!, "Minimum"]), mean(dfGA[!, "Iterations"]), mean(dfGA[!, "f(x) calls"]), String(Symbol(Configs.default_selection)), String(Symbol(Configs.default_crossover)), String(Symbol(m))])

            print("\n\n--------------------------------------------------------------------------------------\n\n")
            print(dfm)
        end
    end

    return dfm

end

n = 50
r = metrics_mutation(n)
CSV.write("mutation_results(n=" * String(Symbol(n)) * ")(time_limit=" * String(Symbol(Configs.time_limit)) * ").csv", r)