abstract Simulator


type POMDPSimulator <: Simulator

  pomdp::POMDP
  
  b::Belief
  s::State
  o::Observation

  trans_dist::AbstractDistribution
  obs_dist::AbstractDistribution

  rng::AbstractRNG
  
  function POMDPSimulator(pomdp::POMDP; rng::AbstractRNG=MersenneTwister(rand(Uint32)))

    b = create_belief(pomdp)
    s = create_state(pomdp)
    o = create_observation(pomdp)

    trans_dist = create_transition_distribution(pomdp)
    obs_dist = create_observation_distribution(pomdp)
    
    return new(pomdp, b, s, o, trans_dist, obs_dist, rng)

  end # function POMDPSimulator

end # type POMDPSimulator


type Exp

  b::Belief
  a::Action
  r::Reward
  bp::Belief

end # type Exp


# Updates simulator state and returns experience
function simulate!(sim::POMDPSimulator, a::Action)
  
  r = reward(sim.pomdp, sim.s, a)

  transition!(sim.trans_dist, sim.pomdp, sim.s, a)
  rand!(sim.rng, sim.s, sim.trans_dist)
  
  observation!(sim.obs_dist, sim.pomdp, sim.s, a)
  rand!(sim.rng, sim.o, sim.obs_dist)

  b = deepcopy(sim.b)
  update_belief!(sim.b, sim.pomdp, a, sim.o)
  bp = deepcopy(sim.b)

  return Exp(b, deepcopy(a), r, bp)

end # function simulate!