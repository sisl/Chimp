typealias ActionSet Vector{Action}


# Interface between simulator and replay dataset
type ExpGain

  deepnet::DeepNet
  simulator::Simulator
  dataset::ReplayDataset
  simover::Bool
  
  actions::ActionSet
  prevBelief::Belief

end # type ExpGain


function select_action(expgain::ExpGain, belief::Belief, epsilon::Float64)

  if rand() < epsilon
    return expgain.actions[rand(1:length(actions))]
  else
    return expgain.actions[select_action(expgain.deepnet, belief)]
  end # if

end # function select_action


function get_epsilon(iter::Int64)

  if iter > Const.EpsilonCount
    return Const.EpsilonMin
  else
    return Const.EpsilonFinal + (Const.EpsilonStart - Const.EpsilonFinal) * 
           max(Const.EpsilonCount - iter, 0) / Const.EpsilonCount
  end # if

end # function get_epsilon


function generate_experience()



end # function generate_experience