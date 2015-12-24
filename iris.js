'use strict';

var fs = require('fs');
var _ = require('underscore');

// https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
var txt = fs.readFileSync('iris.data', 'utf-8');

var classes = {};

var classNameToId = function(name) {
  if (_.has(classes, name)) {
    return classes[name];
  } else {
    var id = _.size(classes);
    classes[name] = id;
    return id;
  }
}

var data =_.chain(txt.split('\n'))
      .map(function(row) { return row.split(','); })
      .filter(function(row) { return row.length > 1; })
      .map(function(row) {
        return row.slice(0, 4).map(parseFloat).concat(classNameToId(row[4]));
      })
      .shuffle()
      .value();

var inputs = _.map(data, function(row) { return row.slice(0, 4); });
var labels = _.map(data, function(row) { return row[4]; });

//console.log(inputs);
//console.log(labels);

fs.writeFileSync('iris_inputs.json', JSON.stringify(inputs));
fs.writeFileSync('iris_labels.json', JSON.stringify(labels));
