typealias Action Any
typealias State Any
typealias Observation Any


abstract Simulator


type POMDPSimulator <: Simulator
    trans_dist::AbstractDistribution
    obs_dist::AbstractDistribution
    b::Belief
    bp::Belief
    s::State
    sp::State
    a::Action
    o::Observation
    r::Float64
    function POMDPSimulator(pomdp::POMDP, b::Belief)
        self = new()
        self.trans_dist = create_transition_distribution(pomdp)
        self.obs_dist = create_observation_distribution(pomdp)
        self.b = deepcopy(b)
        self.bp = deepcopy(b)
        self.s = create_state(pomdp)
        self.sp = create_state(pomdp)
        # This is a hack might need create_action for this
        a = None
        acts = action(pomdp)
        for ap in domain(acts)
            a = ap
        end
        o = create_observation(pomdp)
        self.r = 0.0
        return self
    end
end

function simulate!(sim::POMDPSimulator,
                  pomdp::POMDP,                                                        
                  action::Action,                                                       
                  initial_belief::Belief,
                  initial_state::State;
                  rng=MersenneTwister(rand(Uint32)))
    
    trans_dist = sim.trans_dist
    obs_dist = sim.obs_dist
    sp = sim.sp
    bp = sim.bp
    o = sim.o

    sim.b = initial_belief
    sim.s = initial_state
    sim.a = action

    s = initial_state
    b = initial_belief

    # get the reward
    r = reward(pomdp, s, a)
    sim.r = r

    # get the next state
    transition!(trans_dist, pomdp, s, a)
    rand!(rng, sp, trans_dist)

    # get the observation
    observation!(obs_dist, pomdp, sp, a)
    rand!(rng, o, obs_dist)

    # update the belief
    update_belief!(bp, pomdp, a, o)

    sim
end

function experience_tuple(sim::POMDPSimulator)
    return (sim.b, sim.a, sim.r, sim.bp)
end
