module PODRL

export
    POMDPSimulator,
    simulate!,
    experience_tuple


using POMDPs


include("simulator.jl")

end # module PODRL
