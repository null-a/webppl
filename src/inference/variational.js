'use strict';

var _ = require('underscore');
var numeric = require('numeric');
var Tensor = require('../tensor');
var assert = require('assert');
var util = require('../util.js');
var generic = require('../generic');
var optimize = require('../optimize');
var Histogram = require('../aggregation/histogram');

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

  var getOptimizer, optimizers;

  function Variational(s, k, a, wpplFn, options) {
    var options = util.mergeDefaults(options, {
      steps: 100,
      samplesPerStep: 100,
      returnSamples: 1000,
      optimizer: 'gd',
      miniBatchSize: Infinity,
      throwOnZeroGrad: false,
      callback: function(s, k, a) { return k(s); }
    });

    this.steps = options.steps;
    this.stepSize = options.stepSize;
    this.samplesPerStep = options.samplesPerStep;
    this.returnSamples = options.returnSamples;
    this.miniBatchSize = options.miniBatchSize;
    this.throwOnZeroGrad = options.throwOnZeroGrad;
    this.callback = options.callback;
    this.optimize = getOptimizer(options.optimizer);

    this.curStep = 0;

    this.s = s;
    this.k = k;
    this.a = a;
    this.wpplFn = wpplFn;

    this.coroutine = env.coroutine;
    env.coroutine = this;
  }

  getOptimizer = function(nameOrObj) {
    var name, options;
    if (_.isObject(nameOrObj)) {
      // e.g. { gd: { stepSize: lambda } }
      if (_.size(nameOrObj) !== 1) {
        throw 'Invalid optimizer options.';
      }
      name = _.keys(nameOrObj)[0];
      options = nameOrObj[name];
    } else {
      name = nameOrObj;
      options = {};
    }
    if (!optimize[name]) {
      throw 'Unknown optimizer: ' + name;
    }
    var optimizer = optimize[name](options);
    trace('Will optimize using ' + name + '. ' + JSON.stringify(optimizer.options));
    return optimizer;
  };

  Variational.prototype.run = function() {

    // All variational parameters. Maps addresses to numbers/reals.
    this.params = Object.create(null);

    // Book-keeping for parameter names.
    this.paramPrefixCount = Object.create(null);
    this.paramNames = Object.create(null); // Set of all param names used.
    this.paramAddressNameMap = Object.create(null); // Maps addresses to names.

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

                  _.each(this.paramsSeen, function(val, a) {

                    var g = ad.derivative(val);

                    // L2 regularization.
                    if (_.has(this.regScale, a)) {
                      trace('Computing regularization term for ' + this.paramName(a));
                      g = generic.add(g, generic.scalarMul(ad.value(val), this.regScale[a]));
                    }

                    if (generic.allZero(g)) {
                      var msg = 'Gradient w.r.t parameter ' + this.paramName(a) + ' is zero';
                      if (this.throwOnZeroGrad) {
                        throw msg;
                      } else {
                        info(msg);
                      }
                    }

                    trace('Gradient of objective w.r.t. ' + this.paramName(a) + ':');
                    trace(g);

                    if (!_.has(this.grad, a)) {
                      // Initialize gradients to zero.
                      this.grad[a] = generic.zerosLike(g);
                    }
                    this.grad[a] = generic.add(this.grad[a], g);

                  }, this);

                  return nextSample();
                }.bind(this), this.a);


              }.bind(this),
              function() {

                // * 1/N
                _.each(this.grad, function(g, a) {
                  this.grad[a] = generic.scalarDiv(g, this.samplesPerStep);
                }, this);

                // Take gradient step.
                trace('\n================================================================================');
                trace('Taking gradient step');
                trace('================================================================================\n');
                info('Estimated ELBO before gradient step: ' + this.estELBO);

                trace('Params before step:');
                trace(this.namedParams());

                this.optimize(this.params, this.grad);

                trace('Params after step:');
                debug(this.namedParams());

                env.coroutine = env.defaultCoroutine;
                return this.callback({}, function() {
                  env.coroutine = this;
                  this.curStep += 1;
                  return nextStep();
                }.bind(this), '', this.curStep, this.namedParams());

              }.bind(this));

        }.bind(this),
        this.finish.bind(this));
  };

  Variational.prototype.finish = function() {

    // Reset current step counter for predictable forEach behavior.
    // i.e. The first mini batch is used.
    this.curStep = 0;

    // Build distribution and compute final estimate of ELBO.
    var hist = new Histogram();
    var estELBO = 0;

    return util.cpsLoop(
        this.returnSamples,
        function(i, next) {
          this.logp = 0;
          this.logq = 0;
          this.logr = 0;
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
          trace(this.namedParams());
          env.coroutine = this.coroutine;
          var erp = hist.toERP();
          erp.estELBO = estELBO;
          erp.parameters = this.namedParams();
          return this.k(this.s, erp);
        }.bind(this));
  };

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

  Variational.prototype.registerParamName = function(address, name, prefix) {
    if (!name && !prefix) {
      // No name or prefix given. The stack address will be used
      // (implicitly) as the parameter name.
      return;
    }

    if (!name) {
      // Generate a name from the prefix when name is not given
      // explicitly.
      if (!_.has(this.paramPrefixCount, prefix)) {
        this.paramPrefixCount[prefix] = 0;
      }
      name = prefix + this.paramPrefixCount[prefix];
      this.paramPrefixCount[prefix] += 1;
    }

    if (_.has(this.paramNames, name)) {
      throw 'Parameter ' + name + ' already exists.';
    }
    this.paramNames[name] = true;
    this.paramAddressNameMap[address] = name;
  };

  Variational.prototype.paramName = function(address) {
    return this.paramAddressNameMap[address] || address;
  };

  Variational.prototype.namedParams = function() {
    return _.chain(this.params).map(function(param, a) {
      return [this.paramName(a), param];
    }, this).object().value();
  };

  Variational.prototype.paramChoice = function(s, k, a, erp, params, opts) {
    var options = opts || {};

    if (!_.has(this.params, a)) {
      // New parameter.
      this.registerParamName(a, options.name, options.prefix);

      // Initialize.
      var _val = erp.sample(params);
      this.params[a] = _val;
      debug('Initialized parameter ' + this.paramName(a) + ' to ' + _val);

      if (_.has(opts, 'reg')) {
        assert.ok(opts.reg > 0);
        this.regScale[a] = opts.reg;
        debug('Will regularize parameter ' + this.paramName(a) + ' (Scale = ' + opts.reg + ')');
      }
    } else {
      _val = this.params[a];
      trace('Seen parameter ' + this.paramName(a) + ' before. Value is: ' + _val);
    }
    var val = ad.lift(_val);
    this.paramsSeen[a] = val;
    return k(s, val);
  };

  Variational.prototype.sampleGuide = function(s, k, a, erp, params, opts) {
    // Sample from q.
    // What if a random choice from p is given as a param?

    var options = opts || {};
    var _params = params.map(ad.value);

    var val;
    if ((options.reparam === undefined || options.reparam) &&
        erp.baseParams && erp.transform) {
      // Reparameterization trick.

      // Current params are passed to baseParams so that we can figure
      // out the dimension of multivariate Gaussians. Perhaps this
      // would be nice if we change the ERP interface.

      var baseERP = erp.baseERP || erp;
      var baseParams = erp.baseParams(_params);
      var z = baseERP.sample(baseParams);
      this.logr = ad.scalar.add(this.logr, baseERP.score(baseParams, z));
      val = erp.transform(z, params);
      trace('Sampled ' + ad.value(val) + ' for ' + a);
      trace('  ' + erp.name + '(' + _params + ') reparameterized as ' +
            baseERP.name + '(' + baseParams + ') + transform');
    } else if (options.reparam && !(erp.baseParams && erp.transform)) {
      // Warn when reparameterization is explicitly requested but
      // isn't supported by the ERP.
      throw erp.name + ' ERP does not support reparameterization.';
    } else {
      val = erp.sample(_params);
      this.logr = ad.scalar.add(this.logr, erp.score(params, val));
      trace('Sampled ' + val + ' for ' + a);
      trace('  ' + erp.name + '(' + _params + ')');
    }

    this.logq = ad.scalar.add(this.logq, erp.score(params, val));
    return k(s, val);
  };

  Variational.prototype.forEach = function(s, k, a, arr, f) {

    var m = Math.min(this.miniBatchSize, arr.length);

    if (arr.length % m !== 0) {
      throw 'Mini batch size should be a divisor of total array length (' + arr.length + ').';
    }

    var numBatches = arr.length / m;
    var curBatch = this.curStep % numBatches;
    var miniBatch = arr.slice(curBatch * m, (curBatch + 1) * m);

    var logp0 = this.logp;
    var logq0 = this.logq;
    var logr0 = this.logr;

    return webpplCpsForEach(s, function(s) {
      // Compute score corrections to account for the fact we only
      // looked at a subset of the data.
      var logpdiff = ad.scalar.sub(this.logp, logp0);
      var logqdiff = ad.scalar.sub(this.logq, logq0);
      var logrdiff = ad.scalar.sub(this.logr, logr0);
      this.logp = ad.scalar.add(this.logp, ad.scalar.mul(logpdiff, numBatches - 1));
      this.logq = ad.scalar.add(this.logq, ad.scalar.mul(logqdiff, numBatches - 1));
      this.logr = ad.scalar.add(this.logr, ad.scalar.mul(logrdiff, numBatches - 1));

      return k(s);
    }.bind(this), a.concat('_$' + curBatch), miniBatch, f);
  };

  // Similar to util.cpsForEach but with store/address passing. This
  // is required when f is a webppl function rather than backend CPS
  // code.
  function webpplCpsForEach(s, k, a, arr, f, i) {
    var i = (i === undefined) ? 0 : i;
    if (i === arr.length) {
      return k(s);
    } else {
      return f(s, function(s) {
        return function() {
          return webpplCpsForEach(s, k, a.concat('_$$0'), arr, f, i + 1);
        };
      }, a, arr[i]);
    }
  }

  //FIXME: does this need to trampoline?
  function webpplCpsForEachWithAddresses(s, k, a, arr, add, f, i) {
    var i = (i === undefined) ? 0 : i;
    if (i === arr.length) {
      return k(s);
    } else {
      return f(s, function(s) {
        return function() {
          return webpplCpsForEachWithAddresses(s, k, a, arr, add, f, i + 1);
        };
      }, a.concat('_$$'+add[i]), arr[i]);
    }
  }

  function paramChoice(s, k, a, erp, params, opts) {
    assert.ok(env.coroutine instanceof Variational);
    return env.coroutine.paramChoice(s, k, a, erp, params, opts);
  }

  Variational.prototype.miniBatch = function(s, k, a, arr, f, options) {

    var options = util.mergeDefaults(options, {
      batchSize: 1,
      selectionMethod: 'random'
    });

    var miniBatch=[]
    var batchAddresses=[]
    if(options.selectionMethod == 'random'){
      for(var i=0; i<options.batchSize; i++) {
        var randIndex = Math.floor(Math.random() * arr.length);
        miniBatch.push(arr[randIndex])
        batchAddresses.push(randIndex)
      }
      var numBatches = arr.length / options.batchSize;
    } else if (options.selectionMethod == 'inorder') {
      var m = Math.min(options.batchSize, arr.length);
      if (arr.length % m !== 0) {
        throw 'Mini batch size should be a divisor of total array length (' + arr.length + ').';
      }
      var numBatches = arr.length / m;
      var curBatch = this.curStep % numBatches;
      miniBatch = arr.slice(curBatch * m, (curBatch + 1) * m);
      batchAddresses = _.range(curBatch * m, (curBatch + 1) * m)
    }

    var logp0 = this.logp;
    var logq0 = this.logq;
    var logr0 = this.logr;

    return webpplCpsForEachWithAddresses(s, function(s) {
      // Compute score corrections to account for the fact we only
      // looked at a subset of the data.
      var logpdiff = ad.scalar.sub(this.logp, logp0);
      var logqdiff = ad.scalar.sub(this.logq, logq0);
      var logrdiff = ad.scalar.sub(this.logr, logr0);
      this.logp = ad.scalar.add(this.logp, ad.scalar.mul(logpdiff, numBatches - 1));
      this.logq = ad.scalar.add(this.logq, ad.scalar.mul(logqdiff, numBatches - 1));
      this.logr = ad.scalar.add(this.logr, ad.scalar.mul(logrdiff, numBatches - 1));

      return k(s);
    }.bind(this), a.concat('_$'), miniBatch, batchAddresses, f);
  };

  function sampleGuide(s, k, a, erp, params, transform) {
    assert.ok(env.coroutine instanceof Variational);
    return env.coroutine.sampleGuide(s, k, a, erp, params, transform);
  }

  function forEach(s, k, a, arr, f) {
    assert.ok(env.coroutine instanceof Variational);
    return env.coroutine.forEach(s, k, a, arr, f);
  }

  function miniBatch(s, k, a, arr, f, op) {
    assert.ok(env.coroutine instanceof Variational);
    return env.coroutine.miniBatch(s, k, a, arr, f, op);
  }

  Variational.prototype.incrementalize = env.defaultCoroutine.incrementalize;

  return {
    Variational: function(s, k, a, wpplFn, options) {
      return new Variational(s, k, a, wpplFn, options).run();
    },
    paramChoice: paramChoice,
    sampleGuide: sampleGuide,
    forEach: forEach,
    miniBatch: miniBatch
  };

};
