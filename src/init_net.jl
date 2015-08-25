function init_net(
    beliefs::Array{Float64, 4},
    actions::Array{Float64, 4},
    rewards::Array{Float64, 4},
    nextbeliefs::Array{Float64, 4},
    isterms::Array{Float64, 4})

  # data
  b_data_layer = MemoryDataLayer(
      name="belief",
      tops=[:belief],
      batch_size=MinibatchSize,
      data=Array[beliefs])

  a_r_data_layer = MemoryDataLayer(
      name="action-reward",
      tops=[:action, :reward],
      batch_size=MinibatchSize,
      data=Array[actions, rewards])

  bp_isterm_data_layer = MemoryDataLayer(
      name="next_belief-terminal",
      tops=[:next_belief, :terminal],
      batch_size=MinibatchSize,
      data=Array[nextbeliefs, isterms])

  # TODO: include dropout layers

  # active convnet (Q)
  conv1_actv_layer = ConvolutionLayer(
      )

  pool1_actv_layer = PoolingLayer(
      )

  conv2_actv_layer = ConvolutionLayer(
      )

  pool2_actv_layer = PoolingLayer(
      )

  conv3_actv_layer = ConvolutionLayer(
      )

  pool3_actv_layer = PoolingLayer(
      )

  prod4_actv_layer = InnerProductLayer(
      )

  # snapshot convnet (Qhat)
  conv1_snap_layer = ConvolutionLayer(
      )

  pool1_snap_layer = PoolingLayer(
      )

  conv2_snap_layer = ConvolutionLayer(
      )

  pool2_snap_layer = PoolingLayer(
      )

  conv3_snap_layer = ConvolutionLayer(
      )

  pool3_snap_layer = PoolingLayer(
      )

  prod4_snap_layer = InnerProductLayer(
      )

  # mask-based selection
  # TODO: see ElementWiseLayer

  # loss
  # TODO: see SquareLossLayer

  # return net

end  # init_net