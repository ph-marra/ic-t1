module Configs

using Evolutionary

# ------------- Gerais -------------

mutable struct hypercube
    d::Int64
    a::Int64
    b::Int64
end

function rosenbrock(xs)
    d = length(xs)
    xi = xs[1:(d-1)]
    xnext = xs[2:d]
    return sum(100 .* (xnext .- xi .^ 2) .^ 2 .+ (xi .- 1) .^ 2)
end

time_limit = 30.0
dimensions = [16, 32, 64]
popsizes = [25, 50, 100, 200, 400]

# ------------- GA -------------

minGA = -2048
maxGA = 2048

metsGA= [Evolutionary.AbsDiff(typemin(Float64))]
optsGA= Evolutionary.Options(iterations=typemax(Int64), time_limit=time_limit)

mi = 5 # for uniformranking

selections = [Evolutionary.truncation, Evolutionary.susinv, Evolutionary.best, Evolutionary.sus, Evolutionary.uniformranking(mi)]
# Evolutionary.tournament(ts), Evolutionary.roulette, Evolutionary.rouletteinv

crossovers = [Evolutionary.DC, Evolutionary.AX, Evolutionary.HX, Evolutionary.IC(0.5)]

# Evolutionary.PMX
mutations = [Evolutionary.gaussian(), Evolutionary.uniform()]

default_popsize = popsizes[1]
default_selection = selections[1]
default_crossover = crossovers[1]
default_mutation = mutations[1]

# ------------- PSO -------------

minPSO = -2048
maxPSO = 2048
f_optimum = 0
f_tol = 1e-6

Ns = popsizes
C1s = [2.0, 1.9, 1.8, 1.7]
C2s = [2.0, 2.1, 2.2, 2.3]
omegas = [0.9, 0.85, 0.8]

default_N = popsizes[1]
default_C1 = C1s[1]
default_C2 = C2s[1]
default_omega = omegas[1]

end