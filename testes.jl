include("pso.jl")
include("ga.jl")
include("configs.jl")
using Evolutionary
using Metaheuristics
using Statistics
using DataFrames
using CSV

# --------------- GA --------------------

popsizes = [25, 50, 100, 200, 400]
ts = 2 # tournament_size
mi = 5 # for uniformranking
dimensions = [16, 32, 64]
selections = [Evolutionary.susinv, Evolutionary.truncation, Evolutionary.best, Evolutionary.sus, Evolutionary.uniformranking(mi)]
#Evolutionary.tournament(ts), Evolutionary.roulette, Evolutionary.rouletteinv
crossovers = [Evolutionary.DC, Evolutionary.AX, Evolutionary.HX, Evolutionary.IC(0.5)]
# Evolutionary.PMX
mutations = [Evolutionary.gaussian(), Evolutionary.uniform()]

default_popsize = popsizes[1]
default_selection = selections[2]
default_crossover = crossovers[1]
default_mutation = mutations[1]

d = 16
minGA = -1
maxGA = 1
metsGA= [Evolutionary.AbsDiff(typemin(Float64))]
optsGA= Evolutionary.Options(iterations=typemax(Int64), time_limit=30.0)

hyper = Configs.hypercube(d, minGA, maxGA)
constr = Evolutionary.BoxConstraints(min(hyper.a, hyper.b) .* ones(hyper.d), max(hyper.a, hyper.b) .* ones(hyper.d))
#print(constr)

#dfGA = GA.create_empty_dataframe()
configGA = GA.configsGA(Configs.rosenbrock,
                        hyper,
                        default_popsize, default_selection, default_crossover, default_mutation,
                        optsGA, metsGA)

#r = GA.run_GA(configGA)
#print(r)
#GA.add_collection!(dfGA, configGA, 10)
#print(dfGA)

#print("\n-----------------------------------------------------\n")

# --------------- PSO --------------------

known_pseudomin = 0
d = 32
minPSO = -2048
maxPSO = 2048

dfPSO = PSO.create_empty_dataframe()
configPSO = PSO.configsPSO(Configs.rosenbrock, 1000, 2, 2, 0.9, d,
                           Configs.hypercube(d, minPSO, maxPSO),
                           Information(f_optimum=0),
                           Options(f_calls_limit=typemax(Int64),
                                   iterations=typemax(Int64),
                                   time_limit=5.0,
                                   f_tol=1e-6))


r = PSO.run_PSO(configPSO)
print(r)
PSO.add_collection!(dfPSO, configPSO, 1)
print(dfPSO)