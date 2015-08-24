#= Implements type for reading/writing experiences to the replay dataset.

We assume
(1) Actions and rewards for the full history fit comfortably in memory,
(2) The belief state representation for the full history does not,
(3) A single sample of belief states fits comfortably in memory.

For instance, if the replay dataset stores the last 1 million experiences,
then the history of actions is 1 byte x 1 M = 1 MB. The same holds for the
history of rewards. However, a modest belief state representation might be
a dense vector with a maximum of 1,000 Float64 elements (typical state spaces
are on the order of millions). In this case the full history of 1 million
states would be (1,000 elem x 8 bytes x 1 M = 8 GB).
=#
module ReplayDatasets

push!(LOAD_PATH, ".")

using
  HDF5,
  POMDPs,
  Const,
  Simulators

export
  ReplayDataset,
  close!,
  add_experience!,
  sample,
  sample!


# wrapper around a replay dataset residing on disk as HDF5
type ReplayDataset

  fp::HDF5File
  
  belief::HDF5Dataset
  action::Vector{Action}
  reward::Vector{Reward}
  isterm::Vector{Int32}  # Int32 for Bool; h5 can't use Bool

  rdsize::Int64
  head::Int64  # index of current 'write' location (mimics queue)

  # greatest index of locations with valid (i.e., intialized) experiences;
  # indices in the range [1, self.valid] are OK
  valid::Int64

  function ReplayDataset(
      belief_length::Int64,  # assumes belief is a vector of floats
      filename::String=ReplayDatasetFile,
      overwrite::Bool=false)

    if overwrite
      fp = h5open(ReplayDatasetFile, "w")
    else
      fp = h5open(ReplayDatasetFile, "r+")
    end  # if

    if isvalid(fp)
      
      belief = fp["belief"]
      rdsize = size(belief)[1]
      action = read(fp["action"])
      reward = read(fp["reward"])
      isterm = read(fp["isterm"])

      if rdsize != ReplayDatasetSize
        @printf("[WARN] Dataset loaded from %s is of size %d, not %d as requested. Using existing size.",
                filename, rdsize, ReplayDatasetSize)
      end  # if

    else 

      rdsize = ReplayDatasetSize

      belief = d_create(fp, "belief", datatype(Float64), dataspace(rdsize, belief_length))
      d_create(fp, "action", datatype(Int64), dataspace(rdsize, 1))  # assumes action is the actual action index
      d_create(fp, "reward", datatype(Float64), dataspace(rdsize, 1))
      d_create(fp, "isterm", datatype(Int32), dataspace(rdsize, 1))  # use as booleans

      # TODO: types don't match hdf5dataset
      
      action = zeros(Int64, rdsize)
      reward = zeros(Float64, rdsize)
      isterm = zeros(Int32, rdsize)

      attrs(belief)["head"] = 1
      attrs(belief)["valid"] = 0

    end  # if

    head = read(attrs(belief)["head"])
    valid = read(attrs(belief)["valid"])

    return new(fp, belief, action, reward, isterm, rdsize, head, valid)

  end  # function ReplayDataset

end  # ReplayDataset


# whether f has all the correct datasets
function isvalid(f::HDF5File)

  dnames = names(f)
  
  for name in ["belief", "action", "reward", "isterm"]
    if !(name in dnames)
      return false
    end  #if
  end  # for name
  
  return true

end  # function isvalid


# closes the hdf5 file in |rd|; remember to call, otherwise dataset not saved
function close!(rd::ReplayDataset)

  rd.fp["action"] = rd.action
  rd.fp["reward"] = rd.reward
  rd.fp["isterm"] = rd.isterm
  
  attrs(rd.belief)["head"] = rd.head
  attrs(rd.belief)["valid"] = rd.valid

  close(rd.fp)

end  # function close!


#= Add the next step in a game sequence, i.e. a tuple (b, a, r, bp, isterm)
indicating that at belief |b| the agent took action |a|, received reward |r|,
and *then* ended up in belief, |bp|. |isterm| indicates if it is a terminal
state. The original belief is presumed to be the belief at index (head - 1).

Args:
  action:  index of the action chosen
  reward:  integer value of reward, positive or negative
  belief:  a numpy array of shape NUM_FRAMES x WIDTH x HEIGHT
           or None if this action ended the game.
=#
function add_experience!(rd::ReplayDataset, exp::Exp)

  rd.action[rd.head] = exp.a
  rd.reward[rd.head] = exp.r

  if exp.isterm
    rd.belief[rd.head, :] = exp.bp
    rd.isterm[rd.head] = 1
  else
    rd.isterm[rd.head] = 0
  end  # if

  # update head and valid pointer indices
  rd.head = rd.head % rd.rdsize + 1
  rd.valid = min(rd.rdsize, rd.valid + 1)

end  # function add_experience!


#= Uniformly sample (b,a,r,bp) experiences from the replay dataset.

Args:
    sample_size: (self explanatory)

Returns:
    A tuple of numpy arrays for the |sample_size| experiences 
      
      Exp(b, a, r, bp).

    The first dimension of each array corresponds to the experience
    index. The i_th experience is given by
      
      Exp(belief[i], action[i], reward[i], next_belief[i]).
=#
function sample(rd::ReplayDataset, sample_size::Int64)

  samples = Array(Exp, sample_size)
  sample!(samples, rd)
  return samples

end  # function sample!


# mutating version of sample()
function sample!(samples::Vector{Exp}, rd::ReplayDataset)

  sample_size = length(samples)

  if sample_size >= rd.valid
    ArgumentError(string("Can't draw sample of size ", sample_size, 
                         " from replay dataset of size ", rd.valid))
  end  # if

  indices = rand(1:rd.valid, sample_size)

  # can't include (head - 1) in sample because we don't know the next
  # belief state, so simply resample (very rare if dataset is large)
  while (rd.head - 1) in indices
    indices = rand(1:rd.valid, sample_size)
  end  # while

  sort!(indices)  # increasing order to figure out wrap-around
  next_indices = [index + 1 for index in indices]

  # next state might wrap around end of dataset
  if next_indices[end] == rd.rdsize + 1
    next_indices[end] = 1
  end  # if

  # NOTE: perhaps see replay.py; something interesting was done

  for ibelief in 1:sample_size

    samples[ibelief] = Exp(
        read(rd.belief[indices[ibelief], :]),  # b
        rd.action[next_indices[ibelief]],  # a
        rd.reward[next_indices[ibelief]],  # r
        read(rd.belief[next_indices[ibelief], :],
        rd.isterm[next_indices[ibelief]]))

  end  # for ibelief

end  # function sample!

end  # module ReplayDatasets