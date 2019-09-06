compute_values = (ages, model_parameters, individual_parameters) => {
  if(model_parameters['name'] == 'logistic_parallel') {
    return compute_logistic_parallel(ages, model_parameters['parameters'], individual_parameters)
  }
}

compute_logistic_parallel = (ages, parameters, individual_parameters) => {
  // Model parameters
  g = parameters['g']
  t0 = parameters['tau_mean']
  v0 = Math.exp(parameters['xi_mean'])
  deltas = [0].concat(parameters['deltas'])
  betas = parameters['betas']

  // Individual parameters
  alpha = individual_parameters['alpha']
  tau = individual_parameters['tau']

  // Compute values
  var outputs = []
  var space_shift = 0

  for(var i=0; i<deltas.length; ++i) {
    var output = []
    for(var j=0; j<ages.length; ++j) {
      var r_age = alpha * (ages[j] - t0 - tau);
      var val = - deltas[i] - r_age - space_shift
      output.push(1./(1.+g*Math.exp(val)))
    }
    outputs.push(output)
  }
  return outputs
}
