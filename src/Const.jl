module Const

export MinibatchSize, NetUpdateFreq, DiscountFactor,
       LearnRate, RMSGradMomentum, RMSSqGradMomentum, MinSqGrad,
       EpsilonStart, EpsilonFinal, EpsilonCount,
       ReplayDatasetSize, ReplayStartSize


# deep q-learning
const MinibatchSize = 32
# number of new states generated before deepnet update
const NetUpdateFreq = 100  # increase to 10,000  
const DiscountFactor = 0.99

const Episodes = 50  # increase to 200
const EpisodeLength = 1000  # increase to 100,000

# RMSProp
const LearnRate = 2.5e-4
const RMSGradMomentum = 0.95
const RMSSqGradMomentum = 0.95
const MinSqGrad = 0.01

# epsilon-greedy exploration
const EpsilonStart = 1.0
const EpsilonFinal = 0.1
const EpsilonCount = 500  # increase to 50,000

# replay dataset
const ReplayDatasetSize = 1000  # increase to 1,000,000
const ReplayStartSize = 50  # increase to 50,000
const ReplayDatasetFile = "memory.jld"

end  # module Const