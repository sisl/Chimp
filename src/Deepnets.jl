module Deepnets

push!(LOAD_PATH, ".")

using
  POMDPs,
  Const,
  ReplayDatasets,
  Simulators

export
  Deepnet,
  select_action,
  load_minibatch!,
  update_net!,
  close!


# Mocha setup
if UseGPU
    ENV["MOCHA_USE_CUDA"] = "true"
else
    ENV["MOCHA_USE_NATIVE_EXT"] = "true"
    ENV["OMP_NUM_THREADS"] = 1
    blas_set_num_threads(1)
end # if

using Mocha


# wrapper around deep neural network provided by Mocha.jl
type Deepnet

  # allocate memory for network inputs
  beliefs::Array{Float64, 4}
  actions::Array{Float64, 4}  # indicator on action; e.g., a_1 = [1,0,...,0]
  rewards::Array{Float64, 4}
  nextbeliefs::Array{Float64, 4}
  nonterms::Array{Float64, 4}  # hack: 0 for terminal, gamma for nonterminal
  
  net::Net
  backend::Backend
  solver::Solver

  # TODO: allocate memory for param change
  # TODO: add logger
  # TODO: see if Float32 increases efficiency without sacrificing performance

  function Deepnet(belief_length::Int64, action_length::Int64)

    beliefs = zeros(belief_length, 1, 1, MinibatchSize)
    actions = zeros(action_length, 1, 1, MinibatchSize)
    rewards = zeros(1, 1, 1, MinibatchSize)
    nextbeliefs = zeros(belief_length, 1, 1, MinibatchSize)
    nonterms = zeros(1, 1, 1, MinibatchSize)

    backend = init_backend()
    net = init_net(beliefs, actions, rewards, nextbeliefs, nonterms, backend)
    solver = init_solver()

    return new(
        beliefs,
        actions,
        rewards,
        nextbeliefs,
        nonterms,
        net,
        backend,
        solver)

  end  # function Deepnet

end  # type Deepnet


function init_backend()

  backend = UseGPU ? GPUBackend() : CPUBackend()
  init(backend)

  return backend

end  # function init_backend


# defines network architecture; separate file for organization
include("init_net.jl")


function init_solver()

  # TODO: figure out a way to train only half the network...

  return solver

end  # function init_solver


# returns the index of the action (not actual action or indicator vector)
function select_action(deepnet::Deepnet, belief::Belief)



end  # function select_action


# samples from the replay dataset and writes it to mocha-visible memory
function load_minibatch!(deepnet::Deepnet, rd::ReplayDataset)



end  # function load_minibatch!


function full_pass!(deepnet::Deepnet)



end  # function full_pass!


function forward_pass!(deepnet::Deepnet)



end  # function forward_pass!


# updates deepnet with deepnet.delta
function update_params!(deepnet::Deepnet)



end  # function update_params!


# shuts down Mocha network
function close!(deepnet::Deepnet)

  destroy(deepnet.net)
  shutdown(deepnet.backend)

end  # function close!

end  # module Deepnets