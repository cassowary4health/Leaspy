var ages = [];
var incr_=0.5;
for(var i=60; i<90; i=i+incr_) {
  ages.push(i);
}
var parameters = '';

document.getElementById("file_").onchange = function() {
    var files = document.getElementById('file_').files;

    if (files.length <= 0) {
      return false;
    }

    var fr = new FileReader();

    fr.onload = function(e) {
      console.log(e);
      parameters = JSON.parse(e.target.result);

      var alpha = document.getElementById('rangeAlpha').value;
      var tau = document.getElementById('rangeTau').value;
      individual_parameters = {'alpha': alpha, 'tau': tau}

      data_new = compute_values(ages, parameters, individual_parameters);
      data_new = convert_data_to_plot(ages, data_new)

      Plotly.newPlot('PlotlyTest', data_new);
    }

    fr.readAsText(files.item(0));
}



convert_data_to_plot = (ages, values) => {
  var data = []
  for(var i=0; i<values.length; i++) {
    var y_ = []
    for(var j=0; j<ages.length; ++j) {
      y_.push(values[i][j])
    }
    var trace = {
      x: ages,
      y: y_,
      mode: 'line'
    }
    data.push(trace);
  }
  return data
}

adjustValue = () => {
  var alpha = document.getElementById('rangeAlpha').value;
  var tau = document.getElementById('rangeTau').value;

  individual_parameters = {'alpha': alpha, 'tau': tau}
  up = compute_values(ages, parameters, individual_parameters);

  update = {y:up}

  Plotly.restyle('PlotlyTest', update);
}


adjustValue1 = (alpha) => {

  individual_parameters = {'alpha': alpha, 'tau': 0}
  up = compute_values(ages, parameters, individual_parameters);

  update = {y:up}

  Plotly.restyle('PlotlyTest', update);
}
