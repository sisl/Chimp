
module Deepnets

push!(LOAD_PATH, ".")

using
  Mocha,
  POMDPs,
  Const,
  ReplayDatasets,
  Simulators

export
  Deepnet,
  select_action,
  update_net!


# wrapper around deep neural network provided by Mocha.jl
type Deepnet

  net
  backend

  samples::Vector{Exp}  # allocate memory for network inputs
  delta::Vector{Float64}  # allocate memory for param change

  # TODO: add logger

  function Deepnet()

    samples = Array(Exp, MinibatchSize)
    delta = ???

  end  # function Deepnet

end  # type Deepnet


function copy!(to::Deepnet, from::Deepnet)



end  # function copy!


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