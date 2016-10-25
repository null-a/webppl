// Based on:
// https://github.com/probmods/webppl-daipp/blob/976897167b3abf0acb573718495c6ab7c443f852/techreport/misc.js

// QMR-DT:
// http://www.cs.cmu.edu/afs/cs/project/jair/pub/volume10/jaakkola99a-html/node2.html

'use strict';

var _ = require('underscore');

var pp = function(o) { console.log(JSON.stringify(o, null, 2)); };

var rand = function(r) {
  var a = r[0], b = r[1];
  return a + (Math.random() * (b - a));
};

var randInt = function(r) {
  var a = r[0], b = r[1];
  return _.random(a, b);
};

function sampleQmr(opts) {
  return {
    diseases: _.times(opts.numDiseases, function() {
      return {p: rand(opts.baseProbRange)};
    }),

    symptoms: _.times(opts.numSymptoms, function() {
      return {
        leakProb: rand(opts.leakProbRange),
        parents: _.chain(_.range(opts.numDiseases))
          .shuffle()
          .value()
          .slice(0, randInt(opts.numParentsRange))
          .map(function(i) {
            return {i: i, p: rand(opts.condProbRange)};
          })
      };
    })
  };
};


var qmr = sampleQmr({
  numDiseases: 10,
  numSymptoms: 10,
  numParentsRange: [2, 5],
  baseProbRange: [0, .1],
  leakProbRange: [0, .05],
  condProbRange: [0, .7]
});

pp(qmr);
