module Deepnets

push!(LOAD_PATH, ".")

using POMDPs, Mocha

export Deepnet, select_action


# wrapper around deep neural network provided by Mocha.jl
type Deepnet



end  # type Deepnet


function select_action(deepnet::Deepnet, belief::Belief)



end  # function select_action

end  # module Deepnets