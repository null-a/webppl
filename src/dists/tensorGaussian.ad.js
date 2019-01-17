'use strict';

var _ = require('lodash');
//var ad = require('../ad');
var tf = require('../tf');
var toNumber = require('../tfUtils').toNumber;
var base = require('./base');
var types = require('../types');
var util = require('../util');
var Tensor = require('../tensor');
var numeric = require('../math/numeric');
var gaussian = require('./gaussian');

var LOG_2PI = numeric.LOG_2PI;

function sample(mu, sigma, dims) {
  var buf = tf.buffer(dims);
  var n = buf.size;
  while (n--) {
    buf.values[n] = gaussian.sample(mu, sigma);
  }
  return buf.toTensor();
}

function score(mu, sigma, dims, x) {
  if (!(x instanceof tf.Tensor) || !_.isEqual(x.shape, dims)) {
    return -Infinity;
  }

  var d = x.size;
  var dLog2Pi = d * LOG_2PI;
  var _2dLogSigma = tf.mulStrict(2 * d, tf.log(sigma));
  var sigma2 = tf.pow(sigma, 2);
  var xSubMu = tf.sub(x, mu);
  var z = tf.divStrict(tf.sum(tf.square(xSubMu)), sigma2);

  return tf.mulStrict(-0.5, tf.addN([dLog2Pi, _2dLogSigma, z]));
}

var TensorGaussian = base.makeDistributionType({
  name: 'TensorGaussian',
  desc: 'Distribution over a tensor of independent Gaussian variables.',
  params: [
    {name: 'mu', desc: 'mean', type: types.unboundedReal},
    {name: 'sigma', desc: 'standard deviation', type: types.positiveReal},
    {name: 'dims', desc: 'dimension of tensor', type: types.array(types.positiveInt)}
  ],
  mixins: [base.continuousSupport],
  sample: function() {
    var mu = toNumber(this.params.mu);
    var sigma = toNumber(this.params.sigma);
    var dims = this.params.dims;
    return sample(mu, sigma, dims);
  },
  score: function(x) {
    return score(this.params.mu, this.params.sigma, this.params.dims, x);
  },
  base: function() {
    var dims = this.params.dims;
    return new TensorGaussian({mu: 0, sigma: 1, dims: dims});
  },
  transform: function(x) {
    var mu = this.params.mu;
    var sigma = this.params.sigma;
    return tf.add(tf.mul(x, sigma), mu);
  }
});

module.exports = {
  TensorGaussian: TensorGaussian,
  sample: sample
};
