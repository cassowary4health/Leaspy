addIndividualData = (scores) => {
  var ages = scores['TIME']
  for(var i=0; i < parameters['dimension']; ++i) {
    dataset = {
      label : '',
      data: convertData(ages, scores[String(i)]),
      showLine: false,
      fill: true,
      pointBorderColor: 'rgb(0, 0, 0)',
      pointBackgroundColor: plotColors[i],
      pointRadius: 7
    }

    myChart.data.datasets.push(dataset);
  }

  myChart.update();
}

individualFit = (result) => {
  var indivParameters = result['individual_parameters']
  var scores = result['scores'];

  indivParameters['xi'] = indivParameters['xi'] - parameters['parameters']['xi_mean'];
  indivParameters['tau'] = indivParameters['tau'] - parameters['parameters']['tau_mean'];

  setTriggerValues(indivParameters);
  onTriggerChange();

  addIndividualData(scores);
}

personalize = () => {
  var birthday = document.getElementById('start').value;
  var scores = hot.getData();

  var data = {
    'birthday': birthday,
    'scores': scores,
    'model': parameters
  }

  $.ajax({
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify(data),
    dataType: 'json',
    url: '/',
    success: function (result) {
      individualFit(result);
    }
  });

}
