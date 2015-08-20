module Const

export MinibatchSize, NetUpdateFreq, DiscountFactor,
       LearnRate, RMSGradMomentum, RMSSqGradMomentum, MinSqGrad,
       EpsilonStart, EpsilonFinal, EpsilonCount,
       ReplayDatasetSize, ReplayStartSize

# deep q-learning
const MinibatchSize = 32
const NetUpdateFreq = 10000  # number of new states generated before deepnet update
const DiscountFactor = 0.99  # for Q-learning

# RMSProp
const LearnRate = 2.5e-4
const RMSGradMomentum = 0.95
const RMSSqGradMomentum = 0.95
const MinSqGrad = 0.01

# epsilon-greedy exploration
const EpsilonStart = 1.0
const EpsilonFinal = 0.1
const EpsilonCount = 50000

# replay dataset
const ReplayDatasetSize = 1e6
const ReplayStartSize = 5e4

end # module Const