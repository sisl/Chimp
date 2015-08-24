module Expgains

push!(LOAD_PATH, ".")

using
  POMDPs,
  Const,
  Deepnets,
  ReplayDatasets,
  Simulators

export
  Expgain,
  generate_experience


# training interface with simulator and replay dataset for deepnet
type Expgain

  deepnet::Deepnet
  sim::Simulator
  dataset::ReplayDataset
  actions::Vector{Action}
  
end  # type Expgain


function select_action(expgain::Expgain, belief::Belief, epsilon::Float64)

  if rand() < epsilon
    return expgain.actions[rand(1:length(expgain.actions))]
  else
    return expgain.actions[select_action(expgain.deepnet, belief)]
  end  # if

end  # function select_action


function get_epsilon(iter::Int64)

  if iter > EpsilonCount
    return EpsilonFinal
  else
    return EpsilonFinal + (EpsilonStart - EpsilonFinal) * 
           max(EpsilonCount - iter, 0) / EpsilonCount
  end  # if

end  # function get_epsilon


# mutates simulator in expgain to get new experience
function generate_experience!(expgain::Expgain, iter::Int64)

  a = select_action(expgain, get_epsilon(iter))
  return simulate!(expgain.sim, a)

end  # function generate_experience!

end  # module Expgains