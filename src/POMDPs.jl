module POMDPs

export 
  # concrete types
  Belief,
  State,
  Observation,
  Action,
  Reward,
  
  # abstract types
  POMDP,
  AbstractDistribution,

  # initializers
  create_belief,
  create_observation,
  create_state,
  create_transition_distribution,
  create_observation_distribution,

  # standard mdp functions
  reward,
  transition!,
  observation!,
  update_belief!,
  rand!,  # sample from distribution
  isterminal  # checks if state (not belief) is terminal


typealias Belief Vector{Float64}
typealias State Vector{Float64}
typealias Observation Vector{Float64}
typealias Action Int64
typealias Reward Float64


abstract AbstractDistribution
abstract POMDP


# initializers
create_belief(pomdp::POMDP) = 
  error("$(typeof(b)) does not implement create_belief")

create_state(pomdp::POMDP) =
  error("$(typeof(pomdp)) does not implement create_state")

create_observation(pomdp::POMDP) =
  error("$(typeof(pomdp)) does not implement create_observation")

create_transition_distribution(pomdp::POMDP) = 
  error("$(typeof(pomdp)) does not implement create_transition_distribution")

create_observation_distribution(pomdp::POMDP) =
  error("$(typeof(pomdp)) does not implement create_observation_distribution")

# standard mdp functions
reward(pomdp::POMDP, state::State, action::Action) =
  error("$(typeof(pomdp)) does not implement reward")

transition!(distribution, pomdp::POMDP, state::State, action::Action)  =
  error("$(typeof(pomdp)) does not implement transition!")

observation!(distribution, pomdp::POMDP, state::State, action::Action) =
  error("$(typeof(pomdp)) does not implement observation!")

update_belief!(b::Belief, pomdp::POMDP, bold::Belief, action::Any, obs::Any) =
  error("$(typeof(b)) does not implement update_belief!")

rand!(rng::AbstractRNG, state::Any, d::AbstractDistribution) =
  error("$(typeof(d)) does not implement rand!")

isterminal(pomdp::POMDP, state::State) =
  error("$(typeof(pomdp)) does not implement isterminal")

end  # module POMDPs