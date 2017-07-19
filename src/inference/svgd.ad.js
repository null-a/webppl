// Stein Variational Gradient Descent: A General Purpose Bayesian
// Inference Algorithm
// https://arxiv.org/abs/1608.04471

'use strict';

// This implementation makes the follow assumptions about the model:

// 1. All random choices are continuous.
// 2. All random choices are scalar valued.
// 3. All random choices have unbounded support.
// 4. The structure of the model is fixed. i.e. Every execution of the
// model samples from the same choices in the same order.

// Below, a "particle" is an array of the values sampled during a
// particular execution of the model. i.e. a bare bones execution
// trace.

// Future work:
// 1. Relax above assumptions where possible.
// 2. Support data sub-sampling.
// 3. Implement the heuristic described in the paper for setting the
// kernel width.

var assert = require('assert');
var _ = require('lodash');
var ad = require('../ad');
var util = require('../util');
var optMethods = require('adnn/opt');
var Tensor = require('../tensor');
var CountAggregator = require('../aggregation/CountAggregator');

module.exports = function(env) {

  var initialize = require('./initialize')(env);

  // Check args, create initial particles.
  function entry(s, k, a, model, options) {
    options = util.mergeDefaults(options, {
      steps: 100,
      particles: 10,
      kernelWidth: 1,
      optMethod: 'gd',
      verbose: false
    }, 'SVGD');

    if (!_.isNumber(options.kernelWidth) || options.kernelWidth <= 0) {
      throw new Error('Invalid kernel width');
    }

    var optimizer = util.getValAndOpts(options.optMethod, function(name, opts) {
      name = (name === 'gd') ? 'sgd' : name;
      return optMethods[name](opts);
    });

    var cont = function(particles) {
      return main(s, k, a, model, particles, options.steps, options.kernelWidth, optimizer);
    };

    if (_.isArray(options.particles)) {
      return cont(options.particles);
    } else if (_.isNumber(options.particles)) {
      return initParticles(s, a, model, options.particles, cont);
    } else {
      throw new Error('Expected "particles" to be a number or an array of particles.');
    }
  }

  function main(s, k, a, model, particles, numSteps, kernelWidth, optimizer) {
    var n = particles.length;
    var d = particles[0].length;

    // Main loop.
    return util.cpsLoop(
      numSteps,
      // Body.
      function(i, next) {
        return computeGrads(model, s, a, particles, kernelWidth, function(grads) {
          // Pack particles and their gradients into matrices in order
          // to use the optimization methods in adnn.

          // It's desirable to avoid the overhead of doing this.
          // However, it's better to wait until tensor values random
          // choices are supported before doing so, as things will
          // look different once that change is made.

          // One option would be to pass an array of tensors to the
          // optimization method. Doing so would require scalars to be
          // put inside singleton tensors. (Which incurs overhead for
          // scalar valued things, but at least multivariate stuff is
          // efficient?)

          var gradsMatrix = new Tensor([n, d]).fromArray(grads);
          var particlesMatrix = new Tensor([n, d]).fromArray(particles);
          optimizer(gradsMatrix, particlesMatrix, i); // Modifies particlesMatrix in place.
          particles = particlesMatrix.toArray();
          return next();
        });
      },
      // Continuation.
      function() {
        // Compute return values by replaying the final
        // particles/traces.
        // This currently performs backprop which is unnecessary, and
        // could be removed to save a little work.
        return runAll(model, s, a, particles, function(objs) {
          var agg = new CountAggregator();
          objs.forEach(function(obj) { agg.add(ad.valueRec(obj.val), ad.value(obj.logp)); });
          var dist = agg.toDist();
          dist.particles = particles;
          return k(s, dist);
        });
      });
  }

  // Creates n particles by sampling from the prior.
  function initParticles(s, a, model, n, cont) {
    var particles = [];
    return util.cpsLoop(
      n,
      function(i, next) {
        return initialize(function(trace) {
          particles.push(_.map(trace.choices, 'val'));
          return next();
        }, model, s, env.exit, a, {ad: false});
      },
      function() {
        return cont(particles);
      });
  }

  // Applies symmetric function f to all pairs of xs. Results are
  // returned in a flat array. The helper `ix` can be used to compute
  // the index at which the result of `f(xs[i], xs[j])` will be store
  // in this array.
  function applySymmetricPairwise(xs, f) {
    var out = [];
    var n = xs.length;
    for (var i = 0; i < n; i++) {
      for (var j = i; j < n; j++) {
        out.push(f(xs[i], xs[j], i, j));
      }
    }
    return out;
  }

  function ix(i, j, n) {
    assert.ok(_.isNumber(i));
    assert.ok(_.isNumber(j));
    assert.ok(_.isNumber(n));
    assert.ok(i >= 0 && i < n, 'i is out of range');
    assert.ok(j >= 0 && j < n, 'j is out of range');
    if (i > j) {
      var tmp = i;
      i = j;
      j = tmp;
    }
    return j + (i * (2 * n - (i + 1)) / 2);
  }

  function rbfKernel(x1, x2, h) {
    assert.ok(_.isNumber(h));
    assert.ok(_.isArray(x1));
    assert.ok(_.isArray(x2));
    assert.ok(x1.length === x2.length, 'Length mismatch');
    var d = x1.length;
    var s = 0;
    for (var i = 0; i < d; i++) {
      s += Math.pow(x1[i] - x2[i], 2);
    }
    return Math.exp(-s / h);
  }

  function applyRbfKernelPairwise(particles, h) {
    var kernel = function(pi, pj) {
      return rbfKernel(pi, pj, h);
    };
    return applySymmetricPairwise(particles, kernel);
  }

  // Similar to applyRbfKernelPairwise, but instead of returning an
  // array of results, it returns a function that can be used to
  // conveniently look up result for any pair (i,j).
  function rbfKernelCached(particles, h) {
    var cache = applyRbfKernelPairwise(particles, h);
    var n = particles.length;
    var lookup = function(i, j) {
      return cache[ix(i, j, n)];
    };
    lookup.cache = cache;
    return lookup;
  }

  // Gradient of the RBF kernel w.r.t to x1.
  function rbfKernelGrad(x1, x2, h, k) {
    // k should be the value of the kernel evaluated at (x1,x2,h).
    assert.ok(_.isArray(x1));
    assert.ok(_.isArray(x2));
    assert.ok(_.isNumber(h));
    assert.ok(_.isNumber(k));
    assert.ok(x1.length === x2.length);
    var c = (-2 * k) / h;
    return x1.map(function(x1i, i) {
      return c * (x1i - x2[i]);
    });
  }

  // Could consider caching results and flipping sign of
  // kernelGrad(i,j,h) when computing kernelGrad(j,i,h)?

  function rbfKernelGradByIndex(xs, h, kernel) {
    // kernel is expected to be a function that computes that when
    // applied to (i,j) returns the value of the kernel at (xs[i],
    // xs[j], h).
    assert.ok(_.isArray(xs));
    assert.ok(_.isNumber(h));
    assert.ok(_.isFunction(kernel));
    return function(i, j) {
      return rbfKernelGrad(xs[i], xs[j], h, kernel(i, j));
    };
  }

  function computeKernelAndGrad(particles, h) {
    var kernel = rbfKernelCached(particles, h);
    var kernelGrad = rbfKernelGradByIndex(particles, h, kernel);
    return {
      kernel: kernel,
      kernelGrad: kernelGrad
    };
  }

  // Helpers for working with arrays as vectors.

  // The fact that we have these suggests switching to vectors rather
  // than arrays, but that's probably not a good idea. We will
  // presumably want to support tensor valued random choices
  // eventually, at which point some of the elements in this array
  // will be tensors (of potentially differing size), so a vector
  // won't work.

  // In fact, the arrays in question are effectively traces, so there
  // may be reason to switch from arrays to a more structured trace
  // type at some point.

  function addArr(x, y) {
    assert.ok(_.isArray(x));
    assert.ok(_.isArray(y));
    assert.ok(x.length === y.length);
    return x.map(function(xi, i) {
      return xi + y[i];
    });
  }

  function scaleArr(s, x) {
    assert.ok(_.isNumber(s));
    assert.ok(_.isArray(x));
    return x.map(function(xi) {
      return s * xi;
    });
  }

  // Compute gradients for all particles.
  function computeGrads(model, s0, a0, particles, kernelWidth, cont) {
    return runAll(model, s0, a0, particles, function(objs) {
      var n = particles.length;
      var d = particles[0].length;

      // Kernel and its gradients.
      var ret = computeKernelAndGrad(particles, kernelWidth);
      var kernel = ret.kernel, kernelGrad = ret.kernelGrad;

      // Compute final gradients.
      // Implements the sum in equation 8 from the paper. (For all
      // particles.)
      var zeros = Array(d).fill(0);
      var grads = particles.map(function(particle, i) {
        return objs.reduce(function(acc, obj, j) {
          var logpGrad_j = obj.grad;
          var term = addArr(scaleArr(kernel(i, j), logpGrad_j), kernelGrad(j, i));
          return addArr(acc, term);
        }, zeros);
      });

      // Scale by 1/n as in equation 8. Also flip the sign so that our
      // optimization methods take us in the correct direction.
      var c = -1 / n;
      for (var i in grads) {
        for (var j in grads[i]) {
          grads[i][j] *= c;
        }
      }

      return cont(grads);
    });
  }

  // Compute logp and its gradient for all particles.
  function runAll(model, s0, a0, particles, cont) {
    var objs = [];
    return util.cpsForEach(
      // Body.
      function(particle, i, ps, next) {
        // Use coroutine to compute logp and its gradient for this
        // particle.
        return new runModel(model, particle, s0, a0).run(function(obj) {
          objs.push(obj);
          return next();
        });
      },
      // Continuation.
      function() {
        return cont(objs);
      },
      particles);
  }

  // A coroutine that computes logp(x) and its gradient w.r.t x.
  function runModel(model, particle, s0, a0) {
    assert.ok(_.isFunction(model), 'Model should be a function.');
    assert.ok(_.isArray(particle), 'Particle should be an array.');
    // s0: initial store
    // a0: initial address

    this.model = model;
    this.particle = particle;
    this.s0 = s0;
    this.a0 = a0;

    // State:
    this.logp = 0;
    // Stores (refs to) the lifted random choice values.
    this.choiceNodes = [];
    // Track which choice execution has reached.
    this.currChoice = 0;
  }

  runModel.prototype = {

    run: function(cont) {
      var oldCoroutine = env.coroutine;
      env.coroutine = this;
      return this.model(_.clone(this.s0), function(s, val) {
        if (this.currChoice !== this.particle.length) {
          throw new Error('Something went wrong. There are unconsumed values in the particle/trace.');
        }
        env.coroutine = oldCoroutine;
        this.logp.backprop();
        var grad = this.choiceNodes.map(ad.derivative);
        return cont({val: val, logp: this.logp, grad: grad});
      }.bind(this), this.a0);
    },

    sample: function(s, k, a, dist) {
      'use ad';
      if (!dist.isContinuous) {
        throw new Error('Only continuous distributions are supported.');
      }
      if (this.currChoice >= this.particle.length) {
        throw new Error('Something went wrong. Perhaps the model has structural choices?');
      }
      var _val = this.particle[this.currChoice++];
      if (!_.isNumber(_val)) {
        throw new Error('Expected values in the particle/trace to be numbers.');
      }
      var val = ad.lift(_val);
      this.choiceNodes.push(val);
      this.logp += dist.score(val);
      return k(s, val);
    },

    factor: function(s, k, a, score) {
      'use ad';
      this.logp += score;
      return k(s);
    },

    incrementalize: env.defaultCoroutine.incrementalize,
    constructor: runModel
  };

  return {
    SVGD: entry
  };

};
