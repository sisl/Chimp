function init_snapnet(
    rewards::Array{Float64, 4},
    nextbeliefs::Array{Float64, 4},
    nonterms::Array{Float64, 4},
    nactions::Int64,
    backend::Backend)

  ## define network layers

  # data
  data_layer = MemoryDataLayer(
      name="data",
      batch_size=MinibatchSize,
      data=Array[rewards, nextbeliefs, nonterms],
      tops=[:reward, :next_belief, :nonterm])

  # snapshot convnet
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

  # not actually used; copy params from actv_net
  dummy_loss_layer = SquareLossLayer(
      bottoms=[:target, :nonterm])

  ## construct network
  net = Net(
      "actv_net",
      backend,
      [
          data_layer,
          snap_layers...,
          trgt_layers...,
          dummy_loss_layer])

  return net

end  # function init_snapnet