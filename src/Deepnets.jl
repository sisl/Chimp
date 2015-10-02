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
  targets::Array{Float64, 4}  # computed from snapnet for each episode

  actv_net::Net
  snap_net::Net
  backend::Backend
  solver::Solver

  # TODO: allocate memory for param change
  # TODO: add logger
  # TODO: see if Float32 increases efficiency without sacrificing performance

  function Deepnet(belief_length::Int64, nactions::Int64)

    beliefs = zeros(belief_length, 1, 1, MinibatchSize)
    actions = zeros(nactions, 1, 1, MinibatchSize)
    rewards = zeros(1, 1, 1, MinibatchSize)
    nextbeliefs = zeros(belief_length, 1, 1, MinibatchSize)
    nonterms = zeros(1, 1, 1, MinibatchSize)
    targets = zeros(1, 1, 1, MinibatchSize)

    backend = init_backend()
    actv_net = init_actvnet(beliefs, actions, nactions, backend)
    snap_net = init_snapnet(rewards, nextbeliefs, nonterms, nactions, backend)
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
include("init_actvnet.jl")
include("init_snapnet.jl")


function init_solver()

  exp_dir = "snapshots"

  solver_params = SolverParameters(
      max_iter=1,  # because we only deal with one minibatch each time
      regu_coef=0.0005,
      mom_policy=MomPolicy.Fixed(0.9),
      lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
      load_from=exp_dir)

  solver = SGD(solver_params)

  setup_coffee_lounge(
      solver,
      save_into="exp_dir/statistics.hdf5",
      every_n_iter=1)

  add_coffee_break(
      solver,
      TrainingSummary(),
      every_n_iter=1)

  add_coffee_break(
      solver,
      Snapshot(exp_dir),
      every_n_iter=1)

  return solver

end  # function init_solver


# returns the index of the action (not actual action or indicator vector)
function select_action(deepnet::Deepnet, belief::Belief)



end  # function select_action


# samples from the replay dataset and writes it to mocha-visible memory
function load_minibatch!(deepnet::Deepnet, rd::ReplayDataset)

  # TODO: compute targets for minibatch using snapnet

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