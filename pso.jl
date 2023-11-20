module PSO

import Pkg
Pkg.add("Metaheuristics")
Pkg.add("DataFrames")

include("configs.jl")
using Metaheuristics
using DataFrames

mutable struct configsPSO
    f
    N
    C1
    C2
    omega
    d
    bounds
    info
    opts

    function configsPSO(f, N, C1, C2, omega, d, bounds, info::Information, opts::Options)
        lb = min(bounds.a, bounds.b) .* ones(bounds.d)
        ub = max(bounds.a, bounds.b) .* ones(bounds.d)
        new(f, N, C1, C2, omega, d,
            Metaheuristics.boxconstraints(lb=lb, ub=ub), info, opts)
    end
end

function create_empty_dataframe()
    return DataFrame([[], [], [], [], [], [], [], []], ["Dimension", "Minimum", "Iterations", "f(x) calls", "N", "C1", "C2", "omega"])
end

function run_PSO(config)
    return Metaheuristics.optimize(config.f, config.bounds,
                                   Metaheuristics.PSO(N=config.N, C1=config.C1, C2=config.C2, ω=config.omega,
                                   information=config.info, options=config.opts))
end

# não sei ainda retirar iteracoes e f_calls dessa merda
function add_entry!(df, config, r)
    push!(df, [config.d, Base.minimum(r), r.iteration, r.f_calls, config.N, config.C1, config.C2, config.omega])
end

function add_collection!(df, config, n)
    for i in range(1, n)
        r = run_PSO(config)
        add_entry!(df, config, r)
    end
end

end

