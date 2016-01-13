'use strict';

var _ = require('underscore');
var numeric = require('numeric');
var Tensor = require('../tensor');
var assert = require('assert');
var util = require('../util.js');
var Histogram = require('../aggregation').Histogram;

var logLevel = process.env.LOG_LEVEL ? parseInt(process.env.LOG_LEVEL) : 0;

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
      returnSamples: 1000,
      optimizer: 'gd',
      callback: function(s, k, a) { return k(s); }
    });

    this.steps = options.steps;
    this.stepSize = options.stepSize;
    this.samplesPerStep = options.samplesPerStep;
    this.returnSamples = options.returnSamples;
    this.optimizerName = options.optimizer;
    this.callback = options.callback;

    this.curStep = 0;

    this.s = s;
    this.k = k;
    this.a = a;
    this.wpplFn = wpplFn;

    this.coroutine = env.coroutine;
    env.coroutine = this;
  }

  var optimizers = {
    gd: function(stepSize) {
      return function(params, grad) {
        _.each(grad, function(g, name) {
          assert(_.has(params, name));
          params[name] = sub(params[name], scalarMul(g, stepSize));
        });
      };
    },
    adagrad: function(stepSize) {
      // State.
      // Map from a to running sum of grad^2.
      var g2 = Object.create(null);
      return function(params, grad) {
        _.each(grad, function(g, name) {
          assert(_.has(params, name));
          if (!_.has(g2, name)) {
            // Start with small non-zero g2 to avoid divide by zero.
            g2[name] = scalarMul(onesLike(g), 0.001);
          }
          g2[name] = add(g2[name], mul(g, g));
          params[name] = sub(params[name], scalarMul(div(g, sqrt(g2[name])), stepSize));
        });
      };
    }
  };

  Variational.prototype.run = function() {

    var optimize = optimizers[this.optimizerName](this.stepSize);

    // TODO: Tensor values params?
    // All variational parameters. Maps addresses to numbers/reals.
    this.params = Object.create(null);

    // Maps param names to regularization scaling constant for those
    // parameters for which regularization is requested.
    this.regScale = Object.create(null);

    return util.cpsLoop(
        this.steps,
        function(i, nextStep) {
          trace('\n********************************************************************************');
          info('Step: ' + i);
          trace('********************************************************************************\n');

          // Acuumulate gradients for this step.
          // Maps addresses to gradients.
          this.grad = Object.create(null);

          // Accumulate an estimate of the lower-bound.
          this.estELBO = 0;

          return util.cpsLoop(
              this.samplesPerStep,
              function(j, nextSample) {
                trace('\n--------------------------------------------------------------------------------');
                trace('Sample: ' + j);
                trace('--------------------------------------------------------------------------------\n');

                // Run the program.
                this.logp = 0;
                this.logq = 0; // The log prob of the variational approximation to p
                this.logr = 0; // The log prob of the sampling distribution

                // Params seen this execution.
                // Maps addresses to tapes.
                this.paramsSeen = Object.create(null);

                return this.wpplFn(_.clone(this.s), function(s, val) {

                  this.curStep += 1;

                  trace('Program returned: ' + ad.value(val));
                  trace('logp: ' + ad.value(this.logp));
                  trace('logq: ' + ad.value(this.logq));
                  trace('logr: ' + ad.value(this.logr));

                  var scoreDiff = ad.value(this.logq) - ad.value(this.logp);
                  trace('score diff: ' + scoreDiff);
                  this.estELBO -= scoreDiff / this.samplesPerStep;


                  // Make sure we don't differentiate through scoreDiff in
                  // objective.
                  assert(typeof scoreDiff === 'number');
                  var objective = ad.scalar.sub(
                      ad.scalar.add(
                          ad.scalar.mul(this.logr, scoreDiff),
                          // TODO: Without reparameterization, the expectation
                          // of the following is 0. This is usually simplified
                          // analytically. Optimize?
                          this.logq
                      ),
                      this.logp);

                  objective.backprop();

                  _.each(this.paramsSeen, function(val, name) {

                    var g = ad.derivative(val);

                    // L2 regularization.
                    if (_.has(this.regScale, name)) {
                      trace('Computing regularization term for ' + name);
                      g = add(g, scalarMul(ad.value(val), this.regScale[name]));
                    }

                    trace('Gradient of objective w.r.t. ' + name + ':');
                    trace(g);

                    if (!_.has(this.grad, name)) {
                      // Initialize gradients to zero.
                      this.grad[name] = zerosLike(g);
                    }
                    this.grad[name] = add(this.grad[name], g);

                    // TODO: Reintroduce division by num samples.

                  }, this);

                  return nextSample();
                }.bind(this), this.a);


              }.bind(this),
              function() {

                // * 1/N
                _.each(this.grad, function(g, name) {
                  this.grad[name] = scalarDiv(g, this.samplesPerStep);
                }, this);

                // Take gradient step.
                trace('\n================================================================================');
                trace('Taking gradient step');
                trace('================================================================================\n');
                info('Estimated ELBO before gradient step: ' + this.estELBO);

                trace('Params before step:');
                trace(this.params);

                optimize(this.params, this.grad);

                trace('Params after step:');
                debug(this.params);

                env.coroutine = env.defaultCoroutine;
                return this.callback({}, function() {
                  env.coroutine = this;
                  return nextStep();
                }.bind(this), '', this.curStep, this.params);

              }.bind(this));

        }.bind(this),
        this.finish.bind(this));
  };

  // Polymorphic functions to simplify dealing with scalars and
  // tensors. How much of an overhead would treating all params as
  // Tensors introduce?

  function zerosLike(x) {
    return _.isNumber(x) ? 0 : new Tensor(x.dims);
  }

  function onesLike(x) {
    return _.isNumber(x) ? 1 : new Tensor(x.dims).fill(1);
  }

  function add(a, b) {
    assert.ok(
        _.isNumber(a) && _.isNumber(b) ||
        a instanceof Tensor && b instanceof Tensor);
    return _.isNumber(a) ? a + b : a.add(b);
  }

  function sub(a, b) {
    assert.ok(
        _.isNumber(a) && _.isNumber(b) ||
        a instanceof Tensor && b instanceof Tensor);
    return _.isNumber(a) ? a - b : a.sub(b);
  }

  function mul(a, b) {
    assert.ok(
        _.isNumber(a) && _.isNumber(b) ||
        a instanceof Tensor && b instanceof Tensor);
    return _.isNumber(a) ? a * b : a.mul(b);
  }

  function div(a, b) {
    assert.ok(
        _.isNumber(a) && _.isNumber(b) ||
        a instanceof Tensor && b instanceof Tensor);
    return _.isNumber(a) ? a / b : a.div(b);
  }

  function scalarMul(a, b) {
    assert.ok(_.isNumber(b));
    return _.isNumber(a) ? a * b : a.mul(b);
  }

  function scalarDiv(a, b) {
    assert.ok(_.isNumber(b));
    return _.isNumber(a) ? a / b : a.div(b);
  }

  function sqrt(a) {
    return _.isNumber(a) ? Math.sqrt(a) : a.sqrt();
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
            var scoreDiff = ad.value(this.logq) - ad.value(this.logp);
            estELBO -= scoreDiff / this.returnSamples;
            hist.add(val);
            return next();
          }.bind(this), this.a);
        }.bind(this),
        function() {
          info('\n================================================================================');
          info('Estimated ELBO: ' + estELBO);
          trace('\nOptimized variational parameters:');
          trace(this.params);
          env.coroutine = this.coroutine;
          var erp = hist.toERP();
          erp.estELBO = estELBO;
          erp.parameters = this.params;
          return this.k(this.s, erp);
        }.bind(this));
  };

  function isTape(obj) {
    return _.has(obj, 'sensitivity');
  }

  function resetSensitivities(tape) {
    if (isTape(tape)) {
      tape.sensitivity = 0;
      _.each(tape.tapes, resetSensitivities);
    }
  }

  // TODO: This options arg clashes with the forceSample arg used in MH.
  Variational.prototype.sample = function(s, k, a, erp, params, opts) {
    var options = opts || {};
    // Assume 1-to-1 correspondence between guide and target for now.

    if (!_.has(options, 'guideVal')) {
      throw 'No guide value given';
    }

    // Update log p.
    var val = options.guideVal;
    var _val = ad.value(val);
    trace('Using guide value ' + _val + ' for ' + a + ' (' + erp.name + ')');
    this.logp = ad.scalar.add(this.logp, erp.score(params, val));
    return k(s, val);
  };

  Variational.prototype.factor = function(s, k, a, score) {
    // Update log p.
    this.logp = ad.scalar.add(this.logp, score);
    return k(s);
  };

  Variational.prototype.paramChoice = function(s, k, a, erp, params, opts) {
    var options = opts || {};
    var name = options.name || a;
    if (!_.has(this.params, name)) {
      // New parameter.
      var _val = erp.sample(params);
      this.params[name] = _val;
      debug('Initialized parameter ' + name + ' to ' + _val);

      if (_.has(opts, 'reg')) {
        assert.ok(opts.reg > 0);
        this.regScale[name] = opts.reg;
        debug('Will regularize parameter ' + name + ' (Scale = ' + opts.reg + ')');
      }
    } else {
      _val = this.params[name];
      trace('Seen parameter ' + name + ' before. Value is: ' + _val);
    }
    var val = ad.lift(_val);
    this.paramsSeen[name] = val;
    return k(s, val);
  };

  Variational.prototype.sampleGuide = function(s, k, a, erp, params, opts) {
    // Sample from q.
    // What if a random choice from p is given as a param?

    var options = opts || {};
    var _params = params.map(ad.value);

    var val;
    if (options.reparam) {
      // Reparameterization trick.
      // Requires ERP implement baseParams and transform.
      if (!erp.baseParams || !erp.transform) {
        throw erp.name + ' ERP does not support reparameterization.';
      }
      // Current params are passed to baseParams so that we can figure
      // out the dimension of multivariate Gaussians. Perhaps this
      // would be nice if we change the ERP interface.
      var baseParams = erp.baseParams(_params);
      var z = erp.sample(baseParams);
      this.logr = ad.scalar.add(this.logr, erp.score(baseParams, z));
      val = erp.transform(z, params);
      trace('Sampled ' + ad.value(val) + ' for ' + a);
      trace('  ' + erp.name + '(' + _params + ') reparameterized as ' +
            erp.name + '(' + baseParams + ') + transform');
    } else {
      val = erp.sample(_params);
      this.logr = ad.scalar.add(this.logr, erp.score(params, val));
      trace('Sampled ' + val + ' for ' + a);
      trace('  ' + erp.name + '(' + _params + ')');
    }

    this.logq = ad.scalar.add(this.logq, erp.score(params, val));
    return k(s, val);
  };

  function paramChoice(s, k, a, erp, params, opts) {
    assert.ok(env.coroutine instanceof Variational);
    return env.coroutine.paramChoice(s, k, a, erp, params, opts);
  }

  function sampleGuide(s, k, a, erp, params, transform) {
    assert.ok(env.coroutine instanceof Variational);
    return env.coroutine.sampleGuide(s, k, a, erp, params, transform);
  }

  function getCurStep(s, k, a) {
    assert.ok(env.coroutine instanceof Variational);
    return k(s, env.coroutine.curStep);
  }

  Variational.prototype.incrementalize = env.defaultCoroutine.incrementalize;

  return {
    Variational: function(s, k, a, wpplFn, options) {
      return new Variational(s, k, a, wpplFn, options).run();
    },
    paramChoice: paramChoice,
    sampleGuide: sampleGuide,
    getCurStep: getCurStep
  };

};
