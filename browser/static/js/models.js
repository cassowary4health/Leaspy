compute_values = (ages, model_parameters, individual_parameters) => {
  if(model_parameters['name'] == 'logistic_parallel') {
    return compute_logistic_parallel(ages, model_parameters['parameters'], individual_parameters)
  } else if(model_parameters['name'] == 'logistic') {
    return compute_logistic(ages, model_parameters['parameters'], individual_parameters)
  } else if(model_parameters['name'] == 'linear') {
    return compute_linear(ages, model_parameters['parameters'], individual_parameters)
  }
}

compute_linear = (ages, parameters, individual_parameters) => {
  // Model parameters
  var g = parameters['g']
  var t0 = parameters['tau_mean']
  var v0 = parameters['v0']
  var mixing_matrix = parameters['mixing_matrix']

  // Individual parameters
  var alpha = individual_parameters['alpha']
  var tau = individual_parameters['tau']
  var sources = individual_parameters['sources'];
  var space_shift = undefined;
  if('space_shift' in individual_parameters) {
    space_shift = individual_parameters['space_shift'];
  } else {
    space_shift = math.multiply(mixing_matrix, sources);
  }


  // Compute values
  var outputs = []
  for(var i=0; i<g.length; ++i) {
    var output = []
    for(var j=0; j<ages.length; ++j) {
      var r_age = alpha * (ages[j] - t0 - tau);
      var val = g[i] + space_shift[i] + v0[i] * r_age;
      output.push(val);
    }
    outputs.push(output);
  }
  return outputs;
}



compute_logistic = (ages, parameters, individual_parameters) => {
  // Model parameters
  var g = parameters['g']
  var t0 = parameters['tau_mean']
  var v0 = parameters['v0']
  var mixing_matrix = parameters['mixing_matrix']

  // Individual parameters
  var alpha = individual_parameters['alpha']
  var tau = individual_parameters['tau']
  var sources = individual_parameters['sources']
  var space_shift = math.multiply(mixing_matrix, sources);

  // Compute values
  var outputs = []
  for(var i=0; i<g.length; ++i) {
    var output = []
    for(var j=0; j<ages.length; ++j) {
      g_k = Math.exp(g[i])
      v_k = Math.exp(v0[i])
      var r_age = alpha * (ages[j] - t0 - tau);
      var val = v_k * r_age * (1+g_k) * (1+g_k) / g_k + space_shift[i];
      output.push(1./(1. + g_k * Math.exp(- val)))
    }
    outputs.push(output);
  }
  return outputs;
}


compute_logistic_parallel = (ages, parameters, individual_parameters) => {

  // Model parameters
  var g = Math.exp(parameters['g'])
  var t0 = parameters['tau_mean']
  var v0 = Math.exp(parameters['xi_mean'])
  var deltas = [0].concat(parameters['deltas'])
  var mixing_matrix = parameters['mixing_matrix']

  // Individual parameters
  var alpha = individual_parameters['alpha']
  var tau = individual_parameters['tau']
  var sources = individual_parameters['sources']
  var space_shift = math.multiply(mixing_matrix, sources);

  // Compute values
  var outputs = []
  for(var i=0; i<deltas.length; ++i) {
    var output = []
    for(var j=0; j<ages.length; ++j) {
      var r_age = alpha * v0 * (ages[j] - t0 - tau);
      var val = - deltas[i] - r_age - space_shift[i]
      output.push(1./(1.+g*Math.exp(val)))
    }
    outputs.push(output)
  }
  return outputs
}
