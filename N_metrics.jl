include("pso.jl")
include("ga.jl")
include("configs.jl")
using Evolutionary
using Metaheuristics
using Statistics
using DataFrames
using CSV

# --------------- PSO --------------------

function metrics_N(n)
    dfm = DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "PopSize (N)", "Minimum (avg)", "Iterations (avg)", "f(x) calls (avg)", "C1 (default)", "C2 (default)", "omega (default)"])

    for d in Configs.dimensions
        for p in Configs.Ns
            configPSO = PSO.configsPSO(Configs.rosenbrock,
            p, Configs.default_C1, Configs.default_C2, Configs.default_omega, d,
            Configs.hypercube(d, Configs.minPSO, Configs.maxPSO),
            Information(f_optimum=Configs.f_optimum),
            Options(f_calls_limit=typemax(Int64),
                iterations=typemax(Int64),
                time_limit=Configs.time_limit,
                f_tol=Configs.f_tol))

            dfPSO = PSO.create_empty_dataframe()

            PSO.add_collection!(dfPSO, configPSO, n)

            push!(dfm, [d, p, mean(dfPSO[!, "Minimum"]), mean(dfPSO[!, "Iterations"]), mean(dfPSO[!, "f(x) calls"]), dfPSO[!, "C1"], dfPSO[!, "C2"], dfPSO[!, "omega"]])

            print("\n\n--------------------------------------------------------------------------------------\n\n")
            print(dfm)
        end
    end

    return dfm

end

n = 50
r = metrics_N(n)
CSV.write("N_results(n=" * String(Symbol(n)) * ")(time_limit=" * String(Symbol(Configs.time_limit)) * ").csv", r)