setTriggerValues = (individualParameters) => {
  var acc = Math.exp(individualParameters['xi']);
  document.getElementById('acc_factor').value = acc;

  var tau = individualParameters['tau'];
  document.getElementById('time_shift').value = tau;

  if(!('source_dimension'in parameters)) {
    return;
  }

  var sources = individualParameters['sources'];
  for(var i=0; i<parameters['source_dimension']; ++i) {
    document.getElementById('geom_'+i).value = sources[i];
  };
}


getTriggerValues = () => {
  var values = {
    'alpha': document.getElementById('acc_factor').value,
    'tau': document.getElementById('time_shift').value
  }

  if(!('source_dimension'in parameters)) {
    return;
  }

  var sources = []
  for(var i=0; i<parameters['source_dimension']; ++i) {
    sources.push(document.getElementById('geom_'+i).value)
  }
  values['sources'] = sources;

  return values
}


convertData = (ages, values) => {
  var scatter = []
  for(var i=0; i<ages.length; i++){
    scatter.push({x:ages[i], y:values[i]})
  }
  return scatter;
}

changeTriggerText = (indivParameters) => {
  var acc = indivParameters['alpha'];
  document.getElementById('acc_factor').previousSibling.innerHTML = 'Acceleration factor : ' + acc;

  var time = indivParameters['tau'];
  document.getElementById('time_shift').previousSibling.innerHTML = 'Time shift : ' + time;

  if(!('source_dimension'in parameters)) { return; }

  var sources = indivParameters['sources'];
  for(var i=0; i<parameters['source_dimension']; ++i) {
    document.getElementById('geom_'+i).previousSibling.innerHTML = 'Geometric pattern ' + (i+1) + ' : ' + sources[i];
  }

}



onTriggerChange = () => {
  var indivParameters = getTriggerValues();
  changeTriggerText(indivParameters);
  var values = compute_values(ages, parameters, indivParameters);

  for(var i=0; i<parameters['dimension']; ++i) {
    var data = convertData(ages, values[i])
    myChart.data.datasets[i].data = data;
  }
  myChart.update();
}


addRow = () => {
  hot.alter('insert_row');
}

removeRow = () => {
  hot.alter('remove_row');
}
