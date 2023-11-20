include("pso.jl")
include("ga.jl")
include("configs.jl")
using Evolutionary
using Metaheuristics
using Statistics
using DataFrames
using CSV

# --------------- PSO --------------------

function metrics_cs(n)
    dfm = DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "PopSize (N) (default)", "Minimum (avg)", "Iterations (avg)", "f(x) calls (avg)", "C1", "C2", "omega (default)"])

    for d in Configs.dimensions
        for i in range(1, length(Configs.C1s))
            configPSO = PSO.configsPSO(Configs.rosenbrock,
            Configs.default_N, Configs.C1s[i], Configs.C2s[i], Configs.default_omega, d,
            Configs.hypercube(d, Configs.minPSO, Configs.maxPSO),
            Information(f_optimum=Configs.f_optimum),
            Options(f_calls_limit=typemax(Int64),
                iterations=typemax(Int64),
                time_limit=Configs.time_limit,
                f_tol=Configs.f_tol))

            dfPSO = PSO.create_empty_dataframe()

            PSO.add_collection!(dfPSO, configPSO, n)

            push!(dfm, [d, Configs.default_N, mean(dfPSO[!, "Minimum"]), mean(dfPSO[!, "Iterations"]), mean(dfPSO[!, "f(x) calls"]), dfPSO[!, "C1"], dfPSO[!, "C2"], dfPSO[!, "omega"]])

            print("\n\n--------------------------------------------------------------------------------------\n\n")
            print(dfm)
        end
    end

    return dfm

end

n = 50
r = metrics_cs(n)
CSV.write("cs_results(n=" * String(Symbol(n)) * ")(time_limit=" * String(Symbol(Configs.time_limit)) * ").csv", r)