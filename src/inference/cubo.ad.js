'use strict';

// Implements the reparameterized CUBO estimator described in:
// The χ-Divergence for Approximate Inference
// https://arxiv.org/abs/1611.00328

var _ = require('underscore');
var assert = require('assert');
var fs = require('fs');
var util = require('../util');
var ad = require('../ad');
var paramStruct = require('../paramStruct');
var guide = require('../guide');

module.exports = function(env) {

  function CUBO(wpplFn, s, a, options, state, params, step, cont) {
    this.opts = util.mergeDefaults(options, {
      samples: 1,
      n: 2
    });

    // TODO: Validate n.

    // The current values of all initialized parameters.
    // (Scalars/tensors, not their AD nodes.)
    this.params = params;

    this.step = step;
    this.cont = cont;

    this.wpplFn = wpplFn;
    this.s = s;
    this.a = a;


    env.coroutine = this;
  }

  function maximum(arr) {
    return Math.max.apply(null, arr);
  }

  CUBO.prototype = {

    run: function() {

      var logws = [];
      var grads = [];

      return util.cpsLoop(
        this.opts.samples,

        // Loop body.
        function(i, next) {
          this.iter = i;
          return this.estimateGradient(function(g, obj) {
            grads.push(g);
            logws.push(obj.logw);
            return next();
          });
        }.bind(this),

        // Loop continuation.
        function() {

          var n = this.opts.n;
          var c = maximum(logws);
          var ws = logws.map(function(logw) { return Math.exp(logw - c); });

          var grad = {};
          grads.forEach(function(grad_i, i) {
            paramStruct.mulEq(grad_i, Math.pow(ws[i], n));
            paramStruct.addEq(grad, grad_i);
          });
          paramStruct.mulEq(grad, n / this.opts.samples);

          var cubo = Math.log(util.sum(logws.map(function(logw) {
            return Math.pow(Math.exp(logw), n);
          })) / this.opts.samples) / n;

          env.coroutine = this.coroutine;
          return this.cont(grad, cubo);
        }.bind(this));

    },

    // Compute a single sample estimate of the gradient.

    estimateGradient: function(cont) {
      'use ad';
      // paramsSeen tracks the AD nodes of all parameters seen during
      // a single execution. These are the parameters for which
      // gradients will be computed.
      this.paramsSeen = {};

      this.logp = 0;
      this.logq = 0;

      return this.wpplFn(_.clone(this.s), function() {

        var logweight = this.logp - this.logq;

        if (ad.isLifted(logweight)) { // Handle programs with zero random choices.
          logweight.backprop();
        }

        var grads = _.mapObject(this.paramsSeen, function(params) {
          return params.map(ad.derivative);
        });

        return cont(grads, {logw: ad.value(logweight)});

      }.bind(this), this.a);

    },

    sample: function(s, k, a, dist, options) {
      'use ad';
      options = options || {};

      var guideDist;
      if (options.guide) {
        guideDist = options.guide;
      } else {
        guideDist = guide.independent(dist, a, env);
        if (this.step === 0 &&
            this.opts.verbose &&
            !this.mfWarningIssued) {
          this.mfWarningIssued = true;
          console.log('CUBO: Defaulting to mean-field for one or more choices.');
        }
      }

      var val = this.sampleGuide(guideDist, options);

      this.logp += dist.score(val);
      this.logq += guideDist.score(val);

      return k(s, val);
    },

    sampleGuide: function(dist, options) {
      var val, reparam;
      // TODO: Only reparameterized choices are currently supported.
      if (!(dist.base && dist.transform)) {
        throw dist + ' does not support reparameterization.';
      }
      // Use the reparameterization trick.
      var baseDist = dist.base();
      var z = baseDist.sample();
      val = dist.transform(z);
      return val;
    },

    factor: function(s, k, a, score, name) {
      'use ad';
      if (!isFinite(ad.value(score))) {
        throw new Error('CUBO: factor score is not finite.');
      }
      this.logp += score;
      return k(s);
    },

    incrementalize: env.defaultCoroutine.incrementalize,
    constructor: CUBO

  };

  return function() {
    var coroutine = Object.create(CUBO.prototype);
    CUBO.apply(coroutine, arguments);
    return coroutine.run();
  };

};
