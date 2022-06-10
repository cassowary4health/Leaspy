let addIndividualData = (scores) => {
  var ages = scores['TIME']
  for(var i=0; i < model['dimension']; ++i) {
    var feature_name = model['features'][i]
    var dataset = {
      label: '',
      data: convertData(ages, scores[feature_name]),
      showLine: false,
      fill: true,
      pointBorderColor: 'rgb(0, 0, 0)',
      pointBackgroundColor: PLOT_COLORS[i % PLOT_COLORS.length],
      pointRadius: 7
    }

    myChart.data.datasets.push(dataset);
  }

  myChart.update();
}

let individualFit = (result) => {
  var indivParameters = result['individual_parameters']
  var scores = result['scores'];

  indivParameters['xi'] = indivParameters['xi'] - model['parameters']['xi_mean'];
  indivParameters['tau'] = indivParameters['tau'] - model['parameters']['tau_mean'];

  setTriggerValues(indivParameters);
  onTriggerChange();

  addIndividualData(scores);
}

let personalize = () => {
  var birthday = document.getElementById('start').value;
  var scores = hot.getData();

  var data = {
    'birthday': birthday,
    'scores': scores,
    'model': model,
  }

  $.ajax({
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify(data),
    dataType: 'json',
    url: '/model/personalize',
    success: individualFit
  });

}
