include("pso.jl")
include("ga.jl")
include("configs.jl")
using Evolutionary
using Metaheuristics
using Statistics
using DataFrames
using CSV

# --------------- PSO --------------------

function metrics_omega(n)
    dfm = DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "PopSize (N) (default)", "Minimum (avg)", "Iterations (avg)", "f(x) calls (avg)", "C1 (default)", "C2 (default)", "omega"])

    for d in Configs.dimensions
        for o in Configs.omegas
            configPSO = PSO.configsPSO(Configs.rosenbrock,
            Configs.default_N, Configs.default_C1, Configs.default_C2, o, d,
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
r = metrics_omega(n)
CSV.write("omega_results(n=" * String(Symbol(n)) * ")(time_limit=" * String(Symbol(Configs.time_limit)) * ").csv", r)