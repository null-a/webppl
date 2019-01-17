'use strict';

var _ = require('lodash');
//var ad = require('../ad');
var tf = require('../tf');
var stopGrad = require('../tfUtils').stopGrad;
var base = require('./base');
var types = require('../types');
var util = require('../util');
var Tensor = require('../tensor');
var numeric = require('../math/numeric');
var gaussian = require('./gaussian');
var TensorGaussian = require('./tensorGaussian').TensorGaussian;

var LOG_2PI = numeric.LOG_2PI;

function sample(mu, sigma) {
  var shape = mu.shape;
  var buf = tf.buffer(shape);
  var n = buf.size;
  while (n--) {
    buf.values[n] = gaussian.sample(0, 1);
  }
  var z = buf.toTensor();
  // Here I'm shifting and scaling the whole sample at once, rather
  // than pulling out individual mu and sigma and passing them to
  // `gaussian.sample`. This might be worth doing for adnn too.
  return tf.addStrict(tf.mulStrict(z, sigma), mu);
}

function score(mu, sigma, x) {
  if (!(x instanceof tf.Tensor) || !_.isEqual(x.shape, mu.shape)) {
    return -Infinity;
  }

  var d = mu.size;
  var dLog2Pi = d * LOG_2PI;
  var logDetCov = tf.mulStrict(2, tf.sum(tf.log(sigma)));
  var z = tf.divStrict(tf.subStrict(x, mu), sigma);
  return tf.mul(-0.5, tf.addStrict(
      dLog2Pi, tf.addStrict(
      logDetCov,
      tf.sum(tf.square(z)))));
}

var DiagCovGaussian = base.makeDistributionType({
  name: 'DiagCovGaussian',
  desc: 'A distribution over tensors in which each element is independent and Gaussian distributed, ' +
      'with its own mean and standard deviation. i.e. A multivariate Gaussian distribution with ' +
      'diagonal covariance matrix. The distribution is over tensors that have the same shape as the ' +
      'parameters ``mu`` and ``sigma``, which in turn must have the same shape as each other.',
  params: [
    {name: 'mu', desc: 'mean', type: types.unboundedTensor},
    {name: 'sigma', desc: 'standard deviations', type: types.positiveTensor}
  ],
  mixins: [base.continuousSupport],
  constructor: function() {
    // var _mu = ad.value(this.params.mu);
    // var _sigma = ad.value(this.params.sigma);
    // if (!util.tensorEqDims(_mu, _sigma)) {
    //   throw new Error(this.meta.name + ': mu and sigma should be the same shape.');
    // }
    if (!_.isEqual(this.params.mu.shape, this.params.sigma.shape)) {
      throw new Error(this.meta.name + ': mu and sigma should be the same shape.');
    }
  },
  sample: function() {
    return sample(stopGrad(this.params.mu), stopGrad(this.params.sigma));
  },
  score: function(x) {
    return score(this.params.mu, this.params.sigma, x);
  },
  base: function() {
    var dims = this.params.mu.shape;
    return new TensorGaussian({mu: 0, sigma: 1, dims: dims});
  },
  transform: function(x) {
    var mu = this.params.mu;
    var sigma = this.params.sigma;
    return tf.addStrict(tf.mulStrict(sigma, x), mu);
  }
});

module.exports = {
  DiagCovGaussian: DiagCovGaussian,
  sample: sample,
  score: score
};
