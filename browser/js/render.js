var ages = [];
var incr_=0.5;
for(var i=50; i<90; i=i+incr_) {
  ages.push(i);
}
var parameters = '';

load_plot = (e) => {
  parameters = JSON.parse(e.target.result);

  var alpha = document.getElementById('rangeAlpha').value;
  var tau = document.getElementById('rangeTau').value;

  create_source_trigger(parameters);

  var sources = [];
  for(var i=0; i<parameters['source_dimension']; ++i) {
    var source = document.getElementById('rangeSource'+i);
    sources.push(source.value);
  };
  individual_parameters = {'alpha': alpha, 'tau': tau, 'sources': sources}

  data_new = compute_values(ages, parameters, individual_parameters);
  data_new = convert_data_to_plot(ages, data_new)

  var layout = {
    yaxis: {range: [-0.01, 1.01]}
  };
  Plotly.newPlot('PlotlyTest', data_new, layout, {responsive: true});
}

document.getElementById("file_").onchange = function() {
    var files = document.getElementById('file_').files;

    if (files.length <= 0) {
      return false;
    }

    var fr = new FileReader();
    fr.onload = load_plot;
    fr.readAsText(files.item(0));
}

create_source_trigger = (parameters) => {
  if(!('source_dimension' in parameters)) {
    return
  }
  var number_of_sources = parameters['source_dimension'];
  var col = document.getElementById('geometric_parameters');

  var title = document.createElement('p');
  title.innerText = 'Geometric parameters';
  col.appendChild(title);

  var theta = parameters['parameters'];
  var range = theta['sources_mean'] + theta['sources_std'];

  for(var i=0; i<number_of_sources; ++i) {
    var input = document.createElement('input');
    input.setAttribute('type', 'range');
    input.setAttribute('id', 'rangeSource'+i);
    input.setAttribute('min', -3.*range);
    input.setAttribute('max', 3.*range);
    input.setAttribute('step', 0.1);
    input.setAttribute('value', theta['sources_mean']);
    input.setAttribute('oninput', 'adjustValue()');

    var trigger = document.createElement('form').appendChild(input);
    col.appendChild(trigger);
  }
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


  var sources = [];
  for(var i=0; i<parameters['source_dimension']; ++i) {
    var source = document.getElementById('rangeSource'+i);
    sources.push(source.value);
  };

  var individual_parameters = {'alpha': alpha, 'tau': tau, 'sources': sources}
  var up = compute_values(ages, parameters, individual_parameters);

  displayValues(alpha, tau);

  update = {y:up}
  Plotly.restyle('PlotlyTest', update);
}

displayValues = (alpha, tau) => {
  var acc_title = document.getElementById('acc_title');
  var time_title = document.getElementById('time_title');

  acc_title.innerText = 'Acceleration factor : ' + alpha;
  time_title.innerText = 'Time shift : ' + tau;
};

(function () {
  var alpha = document.getElementById('rangeAlpha').value;
  var tau = document.getElementById('rangeTau').value;
  displayValues(alpha, tau)
})();
