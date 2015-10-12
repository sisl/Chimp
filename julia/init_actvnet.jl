function init_actvnet(
    beliefs::Array{Float64, 4},
    actions::Array{Float64, 4},
    targets::Array{Float64, 4},
    nactions::Int64,
    backend::Backend)

  ## define network layers

  # data
  data_layer = MemoryDataLayer(
      name="data",
      batch_size=MinibatchSize,
      data=Array[beliefs, actions, targets],
      tops=[:belief, :action, :target])

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

  # loss
  loss_layer = SquareLossLayer(
      bottoms=[:target, :predict])

  ## construct network
  net = Net(
      "actv_net",
      backend,
      [
          data_layer,
          actv_layers...,
          pred_layers...,
          loss_layer])

  return net

end  # init_actvnet