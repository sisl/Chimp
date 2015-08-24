module PODRLs

using
  Const,
  Deepnets,
  Expgains,
  ReplayDatasets,
  Simulators

export
  PODRL,
  train!,
  select_action


type PODRL

  pomdp::POMDP
  deepnet::Deepnet
  dataset::ReplayDataset
  sim::Simulator
  actions::Vector{Action}

  function PODRL(pomdp::POMDP)

    return new(
        pomdp,
        Deepnet(),
        ReplayDataset(n_states(pomdp)),
        POMDPSimulator(pomdp),
        actions(pomdp))

  end  # function PODRL

end  # type PODRL


function train!(podrl::PODRL; verbose::Bool=true)

  # TODO: save snapshots every once a while
  # TODO: do stuff to verbose; incorporate logger

  snapnet = Deepnet()

  expgain = Expgain(
      podrl.deepnet,
      podrl.sim,
      podrl.replayDataset,
      podrl.actions)

  for iepoch in 1:Episodes

    copy!(snapnet, podrl.deepnet)
    reset!(podrl.sim)

    for it in 1:EpisodeLength

      exp = generate_experience!(expgain)
      add_experience!(podrl.dataset, exp)
      load_minibatch!(podrl.deepnet, podrl.replayDataset)
      update_delta!(podrl.deepnet, snapnet)
      update!(podrl.deepnet)
      
      if it % NetUpdateFreq == 0
        copy!(snapnet, podrl.deepnet)
      end  # if

    end  # for it
  end  # for iepoch

end  # function train!


# modify deepnet.delta using rmsprop on minibatch gradient computed from snapnet
function update_delta!(deepnet::Deepnet, snapnet::Deepnet)

  grad_qlearn!(deepnet, snapnet)
  grad_rmsprop!(deepnet)

end  # function grad_qlearn!


# compute minibatch gradient using q-learning
function grad_qlearn!(deepnet::Deepnet, snapnet::Deepnet)



end  # function grad_qlearn!


function grad_rmsprop!(deepnet::Deepnet)



end  # function grad_rmsprop!


# wrapper around deepnet select_action
function select_action(podrl::PODRL, belief::Belief)

  return select_action(podrl.deepnet, belief)

end  # function select_action

end  # module PODRLs