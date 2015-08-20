# NOTE: it's dangerous to keep the instance variables that avoid additional
# memory allocation since they point to some object that can be modified
# elsewhere. If anything, we should always reset these pointers to something
# like a null pointer or something, if we want to just avoid memory allocation.
# For instance, we might return two experience tuples (as a type defined here 
# or an actual Julia tuple) that both point to the same thing...

abstract Simulator


type POMDPSimulator <: Simulator

  trans_dist::AbstractDistribution
  obs_dist::AbstractDistribution
  
  # keep these to avoid additional memory allocation
  b::Belief
  bp::Belief
  s::State
  sp::State
  a::Action
  o::Observation
  r::Reward

  function POMDPSimulator(pomdp::POMDP, b::Belief)

    trans_dist = create_transition_distribution(pomdp)
    obs_dist = create_observation_distribution(pomdp)
    
    b = deepcopy(b)
    bp = deepcopy(b)
    
    s = create_state(pomdp)
    sp = create_state(pomdp)

    # TODO: this is a hack might need create_action for this
    # NOTE: I don't get why use the last element in |acts|
    a = None
    acts = action(pomdp)
    for ap in domain(acts)
      a = ap
    end

    o = create_observation(pomdp)
    r = 0.0

    return new(
        trans_dist,  # transition distribution
        obs_dist,  # observation distribution
        b,  # belief state
        bp,  # next belief state
        s,  # state
        sp,  # next state
        a,  # action
        o,  # observation
        r)  # reward

  end # function POMDPSimulator

end # type POMDPSimulator


type Experience

  b::Belief
  a::Action
  r::Reward
  bp::Belief

  function Experience(sim::POMDPSimulator)

    return new(sim.b, sim.a, sim.r, sim.bp)

  end # function Experience

end # type Experience


# Updates simulator state and returns experience
function simulate!(
    sim::POMDPSimulator,
    pomdp::POMDP,                                                        
    action::Action,                                                       
    initial_belief::Belief,
    initial_state::State;
    rng=MersenneTwister(rand(Uint32)))
  
  trans_dict = sim.trans_dist
  obs_dist = sim.obs_dist

  sp = sim.sp
  bp = sim.bp
  o = sim.o

  sim.b = initial_belief
  sim.s = initial_state
  sim.a = action

  s = initial_state
  b = initial_belief

  # update local simulator variables
  # get the reward
  sim.r = reward(pomdp, s, a)

  # get the next state
  transition!(trans_dist, pomdp, s, a)
  rand!(rng, sp, trans_dist)

  # get the observation
  observation!(obs_dist, pomdp, sp, a)
  rand!(rng, o, obs_dist)

  # TODO: add version of |update_belief!| to POMDPs.jl that avoids recreating the distributions
  # update the belief
  update_belief!(bp, pomdp, a, o)

  sim

end # function simulate!


function experience(sim::POMDPSimulator)
  return Experience(sim)
end


function next_state(sim::POMDPSimulator)
  return sim.sp
end
