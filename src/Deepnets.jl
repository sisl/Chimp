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
  actions::Array{Float64, 4}
  rewards::Array{Float64, 4}
  nextbeliefs::Array{Float64, 4}
  isterms::Array{Float64, 4}
  
  net::Net
  backend::Backend

  # TODO: delta::???  # allocate memory for param change

  # TODO: add logger

  function Deepnet(belief_length::Int64)

    beliefs = zeros(belief_length, 1, 1, MinibatchSize)
    actions = zeros(1, 1, 1, MinibatchSize)
    rewards = zeros(1, 1, 1, MinibatchSize)
    nextbeliefs = zeros(belief_length, 1, 1, MinibatchSize)
    isterms = zeros(1, 1, 1, MinibatchSize)

    net = init_net(beliefs, actions, rewards, nextbeliefs, isterms)
    backend = init_backend()
    
    return new(beliefs, actions, rewards, nextbeliefs, isterms, net, backend)

    # TODO: delta = ???

  end  # function Deepnet

end  # type Deepnet


# defines network architecture; separate file for organization
include("init_net.jl")


function init_backend()

  backend = UseGPU ? GPUBackend() : CPUBackend()
  init(backend)

  return backend

end  # function init_backend


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