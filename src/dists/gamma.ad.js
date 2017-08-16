'use strict';

var ad = require('../ad');
var base = require('./base');
var types = require('../types');
var util = require('../util');
var gaussian = require('./gaussian');

// an implementation of Marsaglia & Tang, 2000:
// A Simple Method for Generating Gamma Variables
function sample(shape, scale) {
  if (shape < 1) {
    var r;
    r = sample(1 + shape, scale) * Math.pow(util.random(), 1 / shape);
    if (r === 0) {
      util.warn('gamma sample underflow, rounded to nearest representable support value');
      return Number.MIN_VALUE;
    }
    return r;
  }
  var x, v, u;
  var d = shape - 1 / 3;
  var c = 1 / Math.sqrt(9 * d);
  while (true) {
    do {
      x = gaussian.sample(0, 1);
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    u = util.random();
    if ((u < 1 - 0.331 * x * x * x * x) || (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v)))) {
      return scale * d * v;
    }
  }
}

var Gamma = base.makeDistributionType({
  name: 'Gamma',
  desc: 'Distribution over positive reals.',
  params: [{name: 'shape', desc: '', type: types.positiveReal},
           {name: 'scale', desc: '', type: types.positiveReal}],
  wikipedia: true,
  mixins: [base.continuousSupport],
  sample: function() {
    return sample(ad.value(this.params.shape), ad.value(this.params.scale));
  },
  score: function(x) {
    'use ad';
    var shape = this.params.shape;
    var scale = this.params.scale;
    return (shape - 1) * Math.log(x) - x / scale - ad.scalar.logGamma(shape) - shape * Math.log(scale);
  },
  support: function() {
    return { lower: 0, upper: Infinity };
  },
  // base: function() {
  //   // We only ever sample from this. Could just have a function to
  //   // generate samples?
  //   return new GammaBase({shape: this.params.shape});
  // },
  // transform: function(z) {
  //   'use ad';
  //   // These values are also computed when sampling. Need to reuse
  //   // them. (Not an issue in Pyro, since it has no separate
  //   // transform. (So the sampler is partially differentiable.)
  //   var alpha = this.params.shape;
  //   var d = alpha - 1 / 3;
  //   var c = 1 / Math.sqrt(9 * d);
  //   return d * Math.pow(1 + z * c, 3);
  // },




  // Reparameterization Interface
  // ============================

  // `sampleReparam` is a reparameterized sampler. It is expected to
  // generate a sample from some base distribution, and pass it
  // through a deterministic/differentiable transform.

  // If the base distribution doesn't depend on the parameters we say
  // the distribution is fully parameterized. Otherwise we say the
  // distribution is partially reparameterized.

  // For partially reparameterized distributions we need the ability
  // to compute the score of the base sample, under the base
  // distribution. (e.g. To compute the ELBO.) Since the transform
  // need not be invertable (e.g. Gamma with shape < 1), we can't
  // assume that it is possible to compute this score from transformed
  // sample alone. So, to accommodate this, `sampleReparam` also
  // returns an auxiliary value, from which `scoreBase` is expected to
  // compute the score of the base sample.

  // That is, we expect the following of `sampleReparam` and
  // `scoreBase`:

  // var {val, aux} = dist.sampleReparam(); // val should be a sample from dist
  // score = dist.scoreBase(aux);           // score should be the log density of the base sample

  // A straight forward implementation of this interface would return
  // the value sampled from the base distribution as the auxiliary
  // value. However, we do not restrict the auxiliary value to *only*
  // be the value sampled from the base distribution. This relaxation
  // makes it possible to reuse computation performed in the sampler
  // in the score.



  // Reparameterized accept/reject for Gamma
  // =======================================

  // shape >= 1

  // The base distribution is the Gaussian internal to the
  // accept/reject sampler.

  // shape < 1

  // The base distribution is over the pair (eps, u), where eps is
  // sampled by rejection from a Gaussian (as above), and u is a draw
  // from Uniform(0,1). Since u is independent and uniform on [0,1],
  // the score of the pair is just the score of eps under the accept
  // reject Gaussian. The final value is computed as
  // h(eps)*u^(1/shape). (Where h is the transform performed by the
  // core accept/reject step.)

  // Note that the transform for this case isn't invertible. `u` is
  // tossed out, so it's not possible to map from the final sampled
  // value back to `eps`.

  sampleReparam: function() {
    'use ad';
    var shape = this.params.shape;
    var scale = this.params.scale;
    var obj = sampleReparamCore(shape < 1 ? shape + 1 : shape);
    var z = obj.z;
    var eps = obj.eps;

    var val;
    if (shape < 1) {
      var u = util.random();
      val = z * Math.pow(u, 1 / shape);
    }
    else {
      val = z;
    }
    return {val: val * scale, aux: {z: z}};
  },

  scoreBase: function(aux) {
    'use ad';
    var z = aux.z;
    var shape = this.params.shape;
    var baseShape = shape < 1 ? shape + 1 : shape;

    // The density of the base distribution is arrived at by scaling
    // the target Gamma density by the inverse of the Jacobian of the
    // transform performed by the core Gamma sampler.

    var d = baseShape - 1 / 3;
    return d * Math.log(z) - z - ad.scalar.logGamma(baseShape) - (1 / 6) * Math.log(d);
  }

});

// Core accept/reject sampler for Gamma. Only supports shape >= 1.

// Here, the transformation of the base sample is performed as an AD
// computation from the outset. If the base sample is rejected this
// work done building the graph is wasted, but since the acceptance
// rate is high, this is likely more efficient than only performing
// the AD version of the transform once a sample has been accepted.
// (Since then we'd be performing the forward computation at least
// twice.)

function sampleReparamCore(shape) {
  'use ad';
  if (shape < 1) {
    throw new Error('shape >= 1 expected');
  }
  var x, v, u;
  var d = shape - 1 / 3;
  var c = 1 / Math.sqrt(9 * d);
  while (true) {
    do {
      x = gaussian.sample(0, 1);
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    u = util.random();
    if ((u < 1 - 0.331 * x * x * x * x) || // u and x are unlifted, so could use regular math ops.
        (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v)))) {
      var z = d * v;
      return {z: z, eps: x};
    }
  }
}

module.exports = {
  Gamma: Gamma,
  sample: sample
};
