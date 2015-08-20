module PODRL

using 
  POMDPs,
  HDF5,
  JLD,
  Const
  
include("deepnet.jl")
include("expgain.jl")
include("simulator.jl")
include("replaydataset.jl")

export
    POMDPSimulator,
    simulate!,
    experience_tuple



end # module PODRL