////////////////////////////////////////
// THE MODEL
////////////////////////////////////////

var parameters = ''
var ages = ''
var plotColors = ['rgb(231, 76, 60)', 'rgb(241, 196, 15)', 'rgb(149, 165, 166)', 'rgb(46, 204, 113)']
var types = ['memory', 'concentration', 'praxis', 'language'];

document.getElementById("file_").onchange = function() {
    var files = document.getElementById('file_').files;

    if (files.length <= 0) {
      return false;
    }

    var fr = new FileReader();
    fr.onload = initModel;
    fr.readAsText(files.item(0));
}

triggerInput = (id, value, min, max, step) => {
  var input = document.createElement('input');
  value = Math.round(value/step)*step;
  min = Math.round(min/step)*step;
  max = Math.round(max/step)*step;

  input.setAttribute('type', 'range');
  input.setAttribute('id', id);
  input.setAttribute('value', value);
  input.setAttribute('min', min);
  input.setAttribute('max', max);
  input.setAttribute('step', step);
  input.setAttribute('oninput', 'onTriggerChange()');

  return input
}

triggerCol = (title, id, value, min, max, step) => {
  var title_p = document.createElement('p');
  title_p.innerText = title + ' : ' + value;

  var input = triggerInput(id, value, min, max, step);

  var col = document.createElement('div');
  col.setAttribute('class', 'col-md-12');
  col.appendChild(title_p);
  col.appendChild(input);

  return col
}

initTriggers = (json) => {
  var param = json['parameters'];
  var triggersCol = document.getElementById('triggers');

  // Temporal shift
  var min = - 3 * param['tau_std'];
  var max = + 3 * param['tau_std'];
  var step = 0.5;
  var tempCol = triggerCol('Time shift', 'time_shift', 0, min, max, step);
  triggersCol.appendChild(tempCol);

  // Acceleration factor
  var min = Math.exp(- 3 * param['xi_std']);
  var max = Math.exp(+ 1 * param['xi_std']);
  var step = 0.05;
  var accCol = triggerCol('Acceleration factor', 'acc_factor', 1, min, max, step);
  triggersCol.appendChild(accCol);

  // Space shifts
  if(!('source_dimension'in parameters)) {
    return;
  }

  for(var i=0; i<json['source_dimension']; ++i) {
    var min = - 3 * param['sources_std'];
    var max = + 3 * param['sources_std'];
    var step = 0.1;
    var spaceCol = triggerCol('Geometric pattern '+ (i+1), 'geom_'+i, 0, min, max, step);
    triggersCol.appendChild(spaceCol);
  }
}

clearPage = () => {
  var canvasDiv = document.getElementById("canvas");
  while (canvasDiv.firstChild) {
    canvasDiv.removeChild(canvasDiv.firstChild);
  }

  var canvas = document.createElement('canvas');
  canvas.setAttribute('id', 'myChart');
  canvasDiv.appendChild(canvas);

  var triggers = document.getElementById("triggers");
  while (triggers.firstChild) {
    triggers.removeChild(triggers.firstChild);
  }
}


initPlot = () => {
  var indivParameters = getTriggerValues();
  var incr_=0.5;
  ages = []
  for(var i=35; i<110; i=i+incr_) {
    ages.push(i);
  }

  var data = compute_values(ages, parameters, indivParameters);
  var datasets = []
  for(var j=0; j<2; ++j) {
    var borderWidth = (j==0 ? 3: 1.5)
    var borderDash = (j==0 ? undefined : [8,5])

    for(var i=0; i<data.length; ++i) {
      dataset = {
        label: types[i],
        data: convertData(ages, data[i]),
        fill: 'rgba(0, 0, 0, 0)',
        showLine: true,
        borderDash: borderDash,
        borderWidth: borderWidth,
        borderColor: plotColors[i],
        pointRadius: 0
      }

      datasets.push(dataset);
    }
  }

  var ctx = document.getElementById("myChart");

  myChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: datasets
    },
    options: {
      tooltips: {
        mode: 'index',
        intersect: false,
      },
      hover: {
        mode: 'nearest',
        intersect: true
      },
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero:true
          }
        }],
        xAxes: [{
          ticks: {
            min: 40,
            max: 100
          }
        }],
      },
      animation: {
        duration : 0
      }
    }
  });
}

initModel = (e) => {
  clearPage()
  parameters = JSON.parse(e.target.result);
  initTriggers(parameters);
  initPlot();
}


////////////////////////////////////////
// INITIALIZATION OF NEW PATIENT
////////////////////////////////////////

var hot = '';

resetPatientButton = () => {
  var patient = document.getElementById("patient");
  while (patient.firstChild) {
    patient.removeChild(patient.firstChild);
  }

  patient.innerText = 'Birthday';
  var input = document.createElement('input');
  input.setAttribute('type', 'date');
  input.setAttribute('id', 'start');
  input.setAttribute('name', 'trip-start');
  input.setAttribute('value', '1950-01-01');
  input.setAttribute('min', '1900-01-01');
  input.setAttribute('max', '2000-01-01');
  patient.appendChild(input);

  var addRow = document.createElement('button');
  addRow.setAttribute('type', 'button');
  addRow.setAttribute('class','btn btn-info btn-sm');
  addRow.setAttribute('onclick', 'addRow()');
  addRow.style = 'margin:10px';
  addRow.innerText = 'Add a line';
  patient.appendChild(addRow);

  var removeRow = document.createElement('button');
  removeRow.setAttribute('type', 'button');
  removeRow.setAttribute('class', 'btn btn-warning btn-sm');
  removeRow.setAttribute('onclick', 'removeRow()');
  removeRow.style = 'margin:10px';
  removeRow.innerText = 'Delete last line';
  patient.appendChild(removeRow);

  var personalize = document.createElement('button');
  personalize.setAttribute('type', 'button');
  personalize.setAttribute('class', 'btn btn-success btn-sm');
  personalize.setAttribute('onclick', 'personalize()');
  personalize.style = 'margin:10px';
  personalize.innerText = 'Personalize';
  patient.appendChild(personalize);

  var reset = document.createElement('button');
  reset.setAttribute('type', 'button');
  reset.setAttribute('class', 'btn btn-danger btn-sm');
  //addRow.setAttribute('onclick', 'TODO');
  reset.style = 'margin:10px';
  reset.innerText = 'Reinitialize';
  patient.appendChild(reset);

}

initTable = () => {
  var hotElement = document.querySelector('#table');
  var hotElementContainer = hotElement.parentNode;

  var dataObject = [
    {
      asOf: '08/19/2013',
      val1: 0.15,
      val2: 0.0,
      val3: 0.01,
      val4: 0.05,
    },
    {
      asOf: '08/19/2016',
      val1: 0.25,
      val2: 0.05,
      val3: 0.05,
      val4: 0.15,
    },
    {
      asOf: '08/19/2019',
      val1: 0.5,
      val2: 0.1,
      val3: 0.15,
      val4: 0.3,
    }
  ];

  var hotSettings = {
    licenseKey: 'non-commercial-and-evaluation',
    data: dataObject,
    columns: [
      {
        data: 'asOf',
        type: 'date',
        dateFormat: 'MM/DD/YYYY'
      },
      {
        data: 'val1',
        type: 'numeric',
        numericFormat: {
          pattern: '0.00'
        }
      },
      {
        data: 'val2',
        type: 'numeric',
        numericFormat: {
          pattern: '0.00'
        }
      },
      {
        data: 'val3',
        type: 'numeric',
        numericFormat: {
          pattern: '0.00'
        }
      },
      {
        data: 'val4',
        type: 'numeric',
        numericFormat: {
          pattern: '0.00'
        }
      }
    ],
    stretchH: 'all',
    width: 805,
    autoWrapRow: true,
    maxRows: 22,
    rowHeaders: true,
    colHeaders: [
      'Date',
      'ADAS-1',
      'ADAS-2',
      'ADAS-3',
      'ADAS-4'
    ],
    fillHandle: {
    direction: 'vertical',
    autoInsertRow: true
    }
  };

  hot = new Handsontable(hotElement, hotSettings);
}

initIndividualData = () => {
  resetPatientButton();
  initTable();
}
