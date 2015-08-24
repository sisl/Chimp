
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
  select_action


# wrapper around deep neural network provided by Mocha.jl
type Deepnet

  net
  dataset
  # logger

  # allocate memory for network inputs
  sample_size::Int64
  samples::Vector{Exp}

  function Deepnet(sample_size::Int64=MinibatchSize)

    samples = Array(Exp, sample_size)

  end  # function Deepnet

end  # type Deepnet


function select_action(deepnet::Deepnet, belief::Belief)



end  # function select_action


# samples from the replay dataset and writes it to mocha-visible memory
function load_minibatch!(deepnet::Deepnet)



end  # function load_minibatch!


function full_pass!(deepnet::Deepnet)



end  # function full_pass!


function forward_pass!(deepnet::Deepnet)



end  # function forward_pass!

end  # module Deepnets