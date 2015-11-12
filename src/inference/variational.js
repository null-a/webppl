'use strict';

var _ = require('underscore');
var numeric = require('numeric');
var assert = require('assert');
var util = require('../util.js');
var Histogram = require('../aggregation').Histogram;

var logLevel = parseInt(process.env.LOG_LEVEL) || 2;

function logger(level) {
  if (logLevel >= level) {
    console.log.apply(console, _.toArray(arguments).slice(1));
  }
}

var log = _.partial(logger, 0);
var info = _.partial(logger, 1);
var debug = _.partial(logger, 2);
var trace = _.partial(logger, 3);

module.exports = function(env) {

  function Variational(s, k, a, wpplFn, options) {
    var options = util.mergeDefaults(options, {
      steps: 100,
      stepSize: 0.001,
      samplesPerStep: 100,
      returnSamples: 1000
    });

    this.steps = options.steps;
    this.stepSize = options.stepSize;
    this.samplesPerStep = options.samplesPerStep;
    this.returnSamples = options.returnSamples;

    this.s = s;
    this.k = k;
    this.a = a;
    this.wpplFn = wpplFn;

    this.coroutine = env.coroutine;
    env.coroutine = this;
  }

  Variational.prototype.run = function() {

    //var optimize = gd(this.stepSize);
    var optimize = adagrad(this.stepSize);

    // TODO: Tensor values params?
    // All variational parameters. Maps addresses to numbers/reals.
    this.params = Object.create(null);

    return util.cpsLoop(
      this.steps,
      function(i, nextStep) {
        trace('\n********************************************************************************');
        info('Step: ' + i);
        trace('********************************************************************************\n');

        // Maps param address to estimate of grad of lower bound.
        this.LgradEst = Object.create(null);
        this.meanScoreGrad = Object.create(null);

        // Estimate of optimal (global) control variate is:
        // cvCov / cvVar;
        this.cvCov = 0;
        this.cvVar = 0;

        this.estELBO = 0;

        //log(this.grad);

        return util.cpsLoop(
          this.samplesPerStep,
          function(j, nextSample) {
            trace('\n--------------------------------------------------------------------------------');
            trace('Sample: ' + j);
            trace('--------------------------------------------------------------------------------\n');

            // Run the program.
            this.logp = 0;
            this.logq = 0;
            // Maps addresses to tapes.
            // (Params seen this execution.)
            this.paramsSeen = Object.create(null);

            return this.wpplFn(_.clone(this.s), function(s, val) {
              trace('Program returned: ' + val);
              trace('logp: ' + this.logp);
              trace('logq: ' + ad.untapify(this.logq));

              // Compute gradients.
              ad.yGradientR(this.logq);

              var scoreDiff = ad.untapify(this.logq) - this.logp;
              this.estELBO -= scoreDiff / this.samplesPerStep;

              _.each(this.paramsSeen, function(val, a) {
                // Accumulate gradients.
                if (!_.has(this.LgradEst, a)) {
                  this.LgradEst[a] = 0;
                }
                if (!_.has(this.meanScoreGrad, a)) {
                  this.meanScoreGrad[a] = 0;
                }
                trace('Score gradient w.r.t. ' + a + ': ' + val.sensitivity);

                this.LgradEst[a] += (val.sensitivity * scoreDiff) / this.samplesPerStep;
                this.meanScoreGrad[a] += val.sensitivity / this.samplesPerStep;

                // TODO: Better names, not even Cov now simplified.
                this.cvCov += this.LgradEst[a] * val.sensitivity;
                this.cvVar += Math.pow(val.sensitivity, 2);

              }, this);


              return nextSample();
            }.bind(this), this.a);


          }.bind(this),
          function() {

            // Take gradient step.
            trace('\n================================================================================');
            trace('Taking gradient step');
            trace('================================================================================\n');
            debug('Estimated ELBO before gradient step: ' + this.estELBO);

            // Compute gradient estimate using (global) control variate.
            var grad = Object.create(null);
            var optimalScalar = this.cvCov / this.cvVar;

            trace("Optimal scalar: " + optimalScalar);
            trace('Params before step:');
            trace(this.params);

            _.each(this.LgradEst, function(g, a) {
              assert(_.has(this.meanScoreGrad, a));
              grad[a] = g;// - optimalScalar * this.meanScoreGrad[a];
            }, this);

            optimize(this.params, grad);

            trace('Params after step:');
            debug(this.params);

            return nextStep();
          }.bind(this));

      }.bind(this),
      this.finish.bind(this));
  };


  function gd(stepSize) {
    return function(params, grad) {
      _.each(grad, function(g, a) {
        assert(_.has(params, a));
        params[a] -= stepSize * g;
      });
    };
  }
  
  function adagrad(stepSize) {
    // State.
    // Map from a to running sum of grad^2.
    var g2 = Object.create(null);
    return function(params, grad) {
      _.each(grad, function(g, a) {
        assert(_.has(params, a));
        if (!_.has(g2, a)) {
          g2[a] = 0;
        }
        g2[a] += Math.pow(g, 2);
        params[a] -= (stepSize / Math.sqrt(g2[a])) * g;
      });
    };
  }

  Variational.prototype.finish = function() {
    // Build distribution and compute final estimate of ELBO.
    var hist = new Histogram();
    var estELBO = 0;

    return util.cpsLoop(
      this.returnSamples,
      function(i, next) {
        this.logp = 0;
        this.logq = 0;
        return this.wpplFn(_.clone(this.s), function(s, val) {
          var scoreDiff = ad.untapify(this.logq) - this.logp;
          estELBO -= scoreDiff / this.returnSamples;
          hist.add(val);
          return next();
        }.bind(this), this.a);
      }.bind(this),
      function() {
        info('\n================================================================================');
        info('Estimated ELBO: ' + estELBO);
        info('\nOptimized variational parameters:');
        info(this.params);
        env.coroutine = this.coroutine;
        var erp = hist.toERP();
        erp.estELBO = estELBO;
        erp.parameters = this.params;
        return this.k(this.s, erp);
      }.bind(this));
  };

  // TODO: This options arg clashes with the forceSample arg used in MH.
  Variational.prototype.sample = function(s, k, a, erp, params, options) {
    var options = options || {};
    // Assume 1-to-1 correspondence between guide and target for now.

    var val;

    if (options.paramChoice) {
      if (!_.has(this.params, a)) {
        // New variational parameter.
        var _val = erp.sample(params);
        this.params[a] = _val;
        trace('Initialized parameter ' + a + ' to ' + _val);
      } else {
        _val = this.params[a];
        trace('Seen parameter ' + a + ' before. Value is: ' + _val);
      }
      val = ad.tapify(_val);
      this.paramsSeen[a] = val;
    } else if (options.guideChoice) {
      // Sample from q.
      // Update log q.
      // What if a random choice from p is given as a param?
      var _params = ad.untapify(params);
      val = erp.sample(_params);
      this.logq = ad.add(this.logq, erp.score(params, val));
      trace('Sampled ' + val + ' for ' + a + ' (' + erp.name + ' with params = ' + JSON.stringify(_params) + ')');
    } else if (_.has(options, 'guideVal')) {
      // Update log p.
      val = options.guideVal;
      trace('Using guide value ' + val + ' for ' + a + ' (' + erp.name + ')');
      this.logp += erp.score(params, val);
    } else {
      throw 'No guide value given';
    }

    return k(s, val);
  };

  Variational.prototype.factor = function(s, k, a, score) {
    // Update log p.
    this.logp += score;
    return k(s);
  };

  Variational.prototype.incrementalize = env.defaultCoroutine.incrementalize;

  return {
    Variational: function(s, k, a, wpplFn, options) {
      return new Variational(s, k, a, wpplFn, options).run();
    }
  };

};
