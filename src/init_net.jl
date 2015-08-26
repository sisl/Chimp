function init_net(
    beliefs::Array{Float64, 4},
    actions::Array{Float64, 4},
    rewards::Array{Float64, 4},
    nextbeliefs::Array{Float64, 4},
    nonterms::Array{Float64, 4},
    backend::Backend)

  ## define network layers
  nactions = size(actions, 1)

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

  bp_nonterm_data_layer = MemoryDataLayer(
      name="next_belief-nonterminal",
      tops=[:next_belief, :nonterm],
      batch_size=MinibatchSize,
      data=Array[nextbeliefs, nonterms])

  data_layers = [
      b_data_layer,
      a_r_data_layer,
      bp_nonterm_data_layer]

  # active convnet (Q)
  conv1_actv_layer = ConvolutionLayer(
      name="conv1_actv",
      n_filter=32,
      kernel=(7, 1),
      pad=(3, 1),
      stride=(1, 1),
      filter_init=GaussianInitializer(std=0.01),
      bias_init=ConstantInitializer(0),
      neuron=Neurons.ReLU(),
      bottoms=[:belief],
      tops=[:conv1_actv])

  pool1_actv_layer = PoolingLayer(
      name="pool1_actv",
      kernel=(2, 1),
      stride=(2, 1),
      pooling=Pooling.Max(),
      bottoms=[:conv1_actv],
      tops=[:pool1_actv])

  conv2_actv_layer = ConvolutionLayer(
      name="conv2_actv",
      n_filter=64,
      kernel=(5, 1),
      pad=(2, 1),
      stride=(1, 1),
      filter_init=GaussianInitializer(std=0.01),
      bias_init=ConstantInitializer(1),
      neuron=Neurons.ReLU(),
      bottoms=[:pool1_actv],
      tops=[:conv2_actv])

  pool2_actv_layer = PoolingLayer(
      name="pool2_actv",
      kernel=(2, 1),
      stride=(2, 1),
      pooling=Pooling.Max(),
      bottoms=[:conv2_actv],
      tops=[:pool2_actv])

  conv3_actv_layer = ConvolutionLayer(
      name="conv3_actv",
      n_filter=64,
      kernel=(3, 1),
      pad=(1, 1),
      stride=(1, 1),
      filter_init=GaussianInitializer(std=0.01),
      bias_init=ConstantInitializer(0),
      neuron=Neurons.ReLU(),
      bottoms=[:pool2_actv],
      tops=[:conv3_actv])

  pool3_actv_layer = PoolingLayer(
      name="pool3_actv",
      kernel=(2, 1),
      stride=(2, 1),
      pooling=Pooling.Max(),
      bottoms=[:conv3_actv],
      tops=[:pool3_actv])

  prod4_actv_layer = InnerProductLayer(
      name="prod4_actv",
      output_dim=512,
      weight_init=GaussianInitializer(std=0.005),
      weight_regu=L2Regu(1),
      bias_init=ConstantInitializer(1),
      bottoms=[:pool3_actv],
      tops=[:prod4_actv])

  drop4_actv_layer = DropoutLayer(
      name="drop4_actv",
      ratio=0.5,
      bottoms=[:prod4_actv])

  prod5_actv_layer = InnerProductLayer(
      name="prod5_actv",
      output_dim=nactions,
      weight_init=GaussianInitializer(std=0.01),
      weight_regu=L2Regu(1),
      bias_init=ConstantInitializer(0),
      bottoms=[:prod4_actv],
      tops=[:prod5_actv])

  actv_layers = [
      conv1_actv_layer,
      pool1_actv_layer,
      conv2_actv_layer,
      pool2_actv_layer,
      conv3_actv_layer,
      pool3_actv_layer,
      prod4_actv_layer,
      drop4_actv_layer,
      prod5_actv_layer]

  # snapshot convnet (Qhat)
  conv1_snap_layer = ConvolutionLayer(
      name="conv1_snap",
      n_filter=32,
      kernel=(7, 1),
      pad=(3, 1),
      stride=(1, 1),
      filter_init=GaussianInitializer(std=0.01),
      bias_init=ConstantInitializer(0),
      neuron=Neurons.ReLU(),
      bottoms=[:belief],
      tops=[:conv1_snap])

  pool1_snap_layer = PoolingLayer(
      name="pool1_snap",
      kernel=(2, 1),
      stride=(2, 1),
      pooling=Pooling.Max(),
      bottoms=[:conv1_snap],
      tops=[:pool1_snap])

  conv2_snap_layer = ConvolutionLayer(
      name="conv2_snap",
      n_filter=64,
      kernel=(5, 1),
      pad=(2, 1),
      stride=(1, 1),
      filter_init=GaussianInitializer(std=0.01),
      bias_init=ConstantInitializer(1),
      neuron=Neurons.ReLU(),
      bottoms=[:pool1_snap],
      tops=[:conv2_snap])

  pool2_snap_layer = PoolingLayer(
      name="pool2_snap",
      kernel=(2, 1),
      stride=(2, 1),
      pooling=Pooling.Max(),
      bottoms=[:conv2_snap],
      tops=[:pool2_snap])

  conv3_snap_layer = ConvolutionLayer(
      name="conv3_snap",
      n_filter=64,
      kernel=(3, 1),
      pad=(1, 1),
      stride=(1, 1),
      filter_init=GaussianInitializer(std=0.01),
      bias_init=ConstantInitializer(0),
      neuron=Neurons.ReLU(),
      bottoms=[:pool2_snap],
      tops=[:conv3_snap])

  pool3_snap_layer = PoolingLayer(
      name="pool3_snap",
      kernel=(2, 1),
      stride=(2, 1),
      pooling=Pooling.Max(),
      bottoms=[:conv3_snap],
      tops=[:pool3_snap])

  prod4_snap_layer = InnerProductLayer(
      name="prod4_snap",
      output_dim=512,
      weight_init=GaussianInitializer(std=0.005),
      weight_regu=L2Regu(1),
      bias_init=ConstantInitializer(1),
      bottoms=[:pool3_snap],
      tops=[:prod4_snap])

  drop4_snap_layer = DropoutLayer(
      name="drop4_snap",
      ratio=0.5,
      bottoms=[:prod4_snap])

  prod5_snap_layer = InnerProductLayer(
      name="prod5_snap",
      output_dim=nactions,
      weight_init=GaussianInitializer(std=0.01),
      weight_regu=L2Regu(1),
      bias_init=ConstantInitializer(0),
      bottoms=[:prod4_snap],
      tops=[:prod5_snap])

  snap_layers = [
      conv1_snap_layer,
      pool1_snap_layer,
      conv2_snap_layer,
      pool2_snap_layer,
      conv3_snap_layer,
      pool3_snap_layer,
      prod4_snap_layer,
      drop4_snap_layer,
      prod5_snap_layer]

  # Q(s,a)
  eltw1_actv_layer = ElementWiseLayer(
      name="eltw1_actv",
      operation=ElementWiseFunctors.Multiply(),
      bottoms=[:prod5_actv, :action],
      tops=[:eltw1_actv])

  chan1_actv_layer = ChannelPoolingLayer(
      name="chan1_actv",
      channel_dim=1,
      kernel=nactions,
      stride=1,
      pooling=Pooling.Max(),
      bottoms=[:eltw1_actv],
      tops=[:predict])

  pred_layers = [
      eltw1_actv_layer,
      chan1_actv_layer]

  # r - gamma * max_a' { Qhat(s',a') }
  chan1_snap_layer = ChannelPoolingLayer(
      name="chan1_snap",
      channel_dim=1,
      kernel=nactions,
      stride=1,
      pooling=Pooling.Max(),
      bottoms=[:prod5_snap],
      tops=[:chan1_snap])

  eltw2_snap_layer = ElementWiseLayer(
      name="eltw2_snap",
      operation=ElementWiseFunctors.Multiply(),
      bottoms=[:chan1_snap, :nonterm],
      tops=[:eltw2_snap])

  eltw3_snap_layer = ElementWiseLayer(
      name="eltw3_snap",
      operation=ElementWiseFunctors.Add(),
      bottoms=[:eltw2_snap, :reward],
      tops=[:target])

  trgt_layers = [
      chan1_snap_layer,
      eltw2_snap_layer,
      eltw3_snap_layer]

  # loss
  loss_layer = SquareLossLayer(
      bottoms=[:target, :predict])

  ## construct network
  net = Net(
      "",
      backend,
      [
          data_layers...,
          actv_layers...,
          snap_layers...,
          pred_layers...,
          trgt_layers...,
          loss_layer])

  return net

end  # init_net