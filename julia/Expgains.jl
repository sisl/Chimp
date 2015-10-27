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
    ia = rand(1:length(expgain.actions))
    a = expgain.actions[ia]
    return a, float64(ia)  # deepnet requires Float64 for action index
  else
    ia = select_action(expgain.deepnet, belief)
    a = expgain.actions[ia]
    return a, float64(ia)  # deepnet requires Float64 for action index
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

  a, ia = select_action(expgain, get_epsilon(iter))
  return simulate!(expgain.sim, a, ia)

end  # function generate_experience!

end  # module Expgains