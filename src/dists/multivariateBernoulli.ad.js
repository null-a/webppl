'use strict';

var _ = require('lodash');
var assert = require('assert');
//var ad = require('../ad');
var tf = require('../tf');
var base = require('./base');
var types = require('../types');
var util = require('../util');
var Tensor = require('../tensor');

function mvBernoulliScore(ps, x) {
  // TODO: reinstate for tf.js
  // var _x = ad.value(x);
  // var _ps = ad.value(ps);
  // if (!util.isVector(_x) || !util.tensorEqDim0(_x, _ps)) {
  //   return -Infinity;
  // }

  var xSub1 = tf.sub(x, 1);
  var pSub1 = tf.sub(ps, 1);

  return tf.sum(
      tf.log(
      tf.addStrict(
      tf.mulStrict(x, ps),
      tf.mulStrict(xSub1, pSub1))));
}

var MultivariateBernoulli = base.makeDistributionType({
  name: 'MultivariateBernoulli',
  desc: 'Distribution over a vector of independent Bernoulli variables. Each element ' +
      'of the vector takes on a value in ``{0, 1}``. Note that this differs from ``Bernoulli`` which ' +
      'has support ``{true, false}``.',
  params: [{name: 'ps', desc: 'probabilities', type: types.unitIntervalVector}],
  mixins: [base.finiteSupport],
  sample: function() {
    // TODO: Is there a way to implement this that avoids this (sync.)
    // fetch of the data in the buffer?
    var ps_buf = this.params.ps.buffer();
    var buf = tf.buffer(this.params.ps.shape, 'bool');
    var n = buf.size;
    while (n--) {
      buf.values[n] = util.random() < ps_buf.values[n];
    }
    return buf.toTensor();
  },
  score: function(x) {
    return mvBernoulliScore(this.params.ps, x);
  },
  support: function() {
    var dims = this.params.ps.shape;
    var d = dims[0];
    var n = Math.pow(2, d);
    return _.times(n, function(x) {
      return tf.tensor(toBinaryArray(x, d), dims,  'bool');
    });
  }
});

function toBinaryArray(x, length) {
  assert.ok(x >= 0 && x < Math.pow(2, length));
  var arr = [];
  for (var i = 0; i < length; i++) {
    arr.push(x % 2);
    x = x >> 1;
  }
  return arr;
}

module.exports = {
  MultivariateBernoulli: MultivariateBernoulli
};
