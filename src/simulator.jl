abstract Simulator


type POMDPSimulator <: Simulator

  pomdp::POMDP
  
  b::Belief
  s::State
  o::Observation

  trans_dist::AbstractDistribution
  obs_dist::AbstractDistribution
  rng::AbstractRNG

  simover::Bool
  
  function POMDPSimulator(pomdp::POMDP; rng::AbstractRNG=MersenneTwister(rand(Uint32)))

    b = create_belief(pomdp)  # initial belief based on problem definition
    s = create_state(pomdp)  # initial state based on problem definition
    o = create_observation(pomdp)

    trans_dist = create_transition_distribution(pomdp)
    obs_dist = create_observation_distribution(pomdp)
    
    return new(pomdp, b, s, o, trans_dist, obs_dist, rng, false)

  end  # function POMDPSimulator

end  # type POMDPSimulator


type Exp

  b::Belief
  a::Action
  r::Reward
  bp::Belief
  isterm::Bool  # whether sp is really terminal state

end  # type Exp


function simulate!(sim::POMDPSimulator, a::Action)
  
  r = reward(sim.pomdp, sim.s, a)

  transition!(sim.trans_dist, sim.pomdp, sim.s, a)
  rand!(sim.rng, sim.s, sim.trans_dist)
  
  observation!(sim.obs_dist, sim.pomdp, sim.s, a)
  rand!(sim.rng, sim.o, sim.obs_dist)

  update_belief!(sim.bp, sim.pomdp, sim.b, a, sim.o)

  b = deepcopy(sim.b)
  bp = deepcopy(sim.bp)
  # swap the beliefs so that the b at the next step is bp
  sim.bp, sim.b = sim.b, sim.bp 

  isterm = isterminal(sim.pomdp, sim.s)
  if isterm
    reset!(sim)
  end  # if

  return Exp(bold, deepcopy(a), r, b, isterm)  # must be memory-independent

end  # function simulate!


function reset!(sim::POMDPSimulator)

  sim.b = create_belief(sim.pomdp)
  sim.s = create_state(sim.pomdp)

end  # function reset!
