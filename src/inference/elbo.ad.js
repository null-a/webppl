'use strict';

var _ = require('lodash');
var assert = require('assert');
var fs = require('fs');
var util = require('../util');
//var ad = require('../ad');

var tf = require('@tensorflow/tfjs-core');
var tape = require('@tensorflow/tfjs-core/dist/tape');
var engine = tf.ENV.engine;

var toNumber = require('../tfUtils').toNumber;

var paramStruct = require('../params/struct');
var guide = require('../guide');
var graph = require('./elbograph');

var RootNode = graph.RootNode;
var SampleNode = graph.SampleNode;
var FactorNode = graph.FactorNode;
var SplitNode = graph.SplitNode;
var JoinNode = graph.JoinNode;

module.exports = function(env) {

  function makeELBOEstimator(options) {
    options = util.mergeDefaults(options, {
      samples: 1,
      avgBaselines: true,
      avgBaselineDecay: 0.9,
      // Write a DOT file representation of first graph to disk.
      dumpGraph: false,
      // Use local weight of one for all sample and factor nodes. This
      // is useful in combination with the dumpGraph option for
      // understanding/debugging weight propagation.
      debugWeights: false
    }, 'ELBO');
    return function(wpplFn, s, a, state, step, cont) {
      return new ELBO(wpplFn, s, a, options, state, step, cont).run();
    };
  }

  function ELBO(wpplFn, s, a, options, state, step, cont) {
    this.opts = options;
    this.step = step;
    this.state = state;
    this.cont = cont;

    this.wpplFn = wpplFn;
    this.s = s;
    this.a = a;
    this.guideRequired = true;
    this.isParamBase = true;

    // Initialize mapData state.
    this.mapDataStack = [{multiplier: 1}];
    this.mapDataIx = {};

    // 'state' is a plain JS object provided by Optimize as a means of
    // persisting arbitrary state between calls.
    if (!_.has(this.state, 'baselines')) {
      this.state.baselines = {};
    }
    this.baselineUpdates = {};

    this.oldCoroutine = env.coroutine;
    env.coroutine = this;
  }

  function top(stack) {
    return stack[stack.length - 1];
  }

  // The strategy taken here is to build a graph to (coarsely) track
  // dependency information which we use for variance reduction. This
  // simple approach builds a graph that represents:

  // 1. The order in which random choices are made.
  // 2. The conditional independence information from mapData.

  // This is used to remove some unnecessary (i.e. upstream) terms
  // from the weighting applied to each "grad logq" factor in the LR
  // part of the objective. This improves on the naive implementation
  // which weights each factor by logq - logp of the full execution.

  // The graph is built as the program executes. A separate pass then
  // propagates weights back up the graph, taking account of any
  // sub-sampling of data that happens at mapData. After this pass, a
  // node's weight property is the correct weighting for the
  // corresponding grad logq factor of the objective.

  function checkScoreIsFinite(score, source) {
    //var _score = ad.value(score);
    var _score = toNumber(score);
    if (!isFinite(_score)) { // Also catches NaN.
      var msg = 'ELBO: The score of the previous sample under the ' +
          source + ' program was ' + _score + '.';
      if (_.isNaN(_score)) {
        msg += ' Reducing the step size may help.';
      }
      throw new Error(msg);
    }
  }

  ELBO.prototype = {

    run: function() {

      var elbo = 0;
      var grad = {};

      return util.cpsLoop(
          this.opts.samples,

          // Loop body.
          function(i, next) {
            //console.log('ELBO - loop body');
            this.iter = i;
            return this.estimateGradient(function(g, elbo_i) {
              paramStruct.addEq(grad, g); // Accumulate gradient estimates.
              elbo += elbo_i;
              return next();
            });
          }.bind(this),

          // Loop continuation.
          function() {
            //console.log('ELBO - loop continuation');
            // TODO: reinstate support for multiple samples
            assert.ok(this.opts.samples === 1);
            //paramStruct.divEq(grad, this.opts.samples);
            //elbo /= this.opts.samples;
            // TODO: reinstate baselines
            assert.ok(!this.opts.avgBaselines);
            //this.updateBaselines();
            env.coroutine = this.oldCoroutine;
            return this.cont(grad, elbo);
          }.bind(this));

    },

    // Compute a single sample estimate of the gradient.

    estimateGradient: function(cont) {

      //console.log('ELBO - estimate gradient');

      // paramsSeen tracks the AD nodes of all parameters seen during
      // a single execution. These are the parameters for which
      // gradients will be computed.
      this.paramsSeen = {};

      // This tracks nodes as we encounter them which saves doing a
      // topological sort later on.
      this.nodes = [];

      var root = new RootNode();
      this.prevNode = root; // prevNode becomes the parent of the next node.
      this.nodes.push(root);

      //console.log(tf.memory());

      engine.startScope('gradients', true);

      return this.wpplFn(_.clone(this.s), function() {

        graph.propagateWeights(this.nodes); // propagates regular values no graph built here

        // if (this.step === 0 && this.iter === 0 && this.opts.dumpGraph) {
        //   // To vizualize with Graphviz use:
        //   // dot -Tpng -O deps.dot
        //   var dot = graph.generateDot(this.nodes);
        //   fs.writeFileSync('deps.dot', dot);
        // }

        var ret = this.buildObjective(); // need to back prop through this.

        if (typeof(ret.objective) === 'number') {
          // it's possible that the object doesn't depend on the
          // parameters, and is just a number. in such a case we turn
          // the objective into a tensor in order to avoid special
          // casing below.
          ret.objective = tf.scalar(ret.objective);
        }


        var accumulatedGradientMap = {};
        accumulatedGradientMap[ret.objective.id] = tf.ones([]); // we know the objective is a scalar, hence []

        // TODO: filter the active tape? (whatever that does...)
        tape.backpropagateGradients(accumulatedGradientMap, engine.activeTape);
        //console.log(accumulatedGradientMap);

        // if (ad.isLifted(ret.objective)) { // Handle programs with zero random choices.
        //   ret.objective.backprop();
        // }

        var grads = _.mapValues(this.paramsSeen, function(param, name) {
          // param will be a tf.Variable
          if (_.has(accumulatedGradientMap, param.id)) {
            return accumulatedGradientMap[param.id];
          }
          else {
            // TODO: handle this. (happens when param is not used.)
            // options include:
            // * just drop this key from `grads` (seems like the best option)
            // * return zero tensor
            // * return some other value that indicates zero. (but then optimisers have to handle)
            throw 'no grad found for param "' + name + '"';
          }
        });

        //console.log(tf.memory());
        engine.endScope({grads}, true); // pass grads here so that they aren't disposed
        //console.log(tf.memory());

        //grads.mu.print(); // is no disposed
        //this.paramsSeen.mu.print(); // neither is this

        //console.log('ELBO - done estimate gradient');

        return cont(grads, -ret.negElbo);


      }.bind(this), this.a);

    },

    buildObjective: function() {
      'use ad';
      var rootNode = this.nodes[0];
      assert.ok(rootNode instanceof RootNode);
      assert.ok(_.isNumber(rootNode.weight));

      var objective = this.nodes.reduce(function(acc, node) {
        if (node instanceof SampleNode && node.reparam) {
          return acc + node.multiplier * (node.logq - node.logp);
        } else if (node instanceof SampleNode) {
          var weight = node.weight;
          assert.ok(_.isNumber(weight));
          var b = this.computeBaseline(node.address, weight);
          return acc + node.multiplier * ((node.logq * (weight - b)) - node.logp);
        } else if (node instanceof FactorNode) {
          return acc - node.multiplier * node.score;
        } else {
          return acc;
        }
      }.bind(this), 0);
      var elbo = -rootNode.weight;
      var negElbo = rootNode.weight; // i'm returning this for now, to avoid turning the js number `root.weight` into a rank 0 tensor
      return {objective: objective, elbo: elbo, negElbo};
    },

    computeBaseline: function(address, weight) {
      if (!this.opts.avgBaselines) {
        return 0;
      }

      var baselines = this.state.baselines;
      var baselineUpdates = this.baselineUpdates;

      // Accumulate the mean of the weights for each factor across
      // all samples taken this step. These are incorporated into
      // the running average once all samples have been taken.
      // Note that each factor is not necessarily encountered the
      // same number of times.

      if (!_.has(baselineUpdates, address)) {
        baselineUpdates[address] = {n: 1, mean: weight};
      } else {
        var prev = baselineUpdates[address];
        var n = prev.n + 1;
        var mean = (prev.n * prev.mean + weight) / n;
        baselineUpdates[address].n = n;
        baselineUpdates[address].mean = mean;
      }

      // During the first step we'd like to use the weight as the
      // baseline. The hope is that this strategy might avoid very
      // large gradients on the first step. If the initial baseline
      // was zero, these large gradients may cause optimization
      // methods with adaptive step sizes to reduce the step size (for
      // associated parameters) more than will be necessary once the
      // baseline takes effect. This might slow the initial phase of
      // optimization. However, using exactly the weight would cause
      // the gradient to be zero which in turn would trigger a warning
      // from Optimize. To avoid this we scale the weight and use that
      // as the initial baseline.

      return _.has(baselines, address) ? baselines[address] : weight * .99;
    },

    updateBaselines: function() {
      var decay = this.opts.avgBaselineDecay;
      var baselines = this.state.baselines;
      // Note that this leaves untouched the estimate of the average
      // weight for any factors not seen during this step.
      _.each(this.baselineUpdates, function(obj, address) {
        baselines[address] = _.has(baselines, address) ?
            decay * baselines[address] + (1 - decay) * obj.mean :
            obj.mean;
      }, this);
    },

    sample: function(s, k, a, dist, options) {
      options = options || {};
      return guide.getDist(options.guide, options.noAutoGuide, dist, env, s, a, function(s, guideDist) {
        if (!guideDist) {
          throw new Error('ELBO: No guide distribution to optimize.');
        }

        var ret = this.sampleGuide(guideDist, options);
        var val = ret.val;

        var logp = dist.score(val);
        var logq = guideDist.score(val);
        checkScoreIsFinite(logp, 'target');
        checkScoreIsFinite(logq, 'guide');

        var m = top(this.mapDataStack).multiplier;

        var node = new SampleNode(
            this.prevNode, logp, logq,
            ret.reparam, a, dist, guideDist, val, m, this.opts.debugWeights);

        this.prevNode = node;
        this.nodes.push(node);

        return k(s, val);

      }.bind(this));
    },

    sampleGuide: function(dist, options) {
      var val, reparam;

      if ((!_.has(options, 'reparam') || options.reparam) &&
          dist.base && dist.transform) {
        // Use the reparameterization trick.
        var baseDist = dist.base();
        var z = baseDist.sample();
        val = dist.transform(z);
        reparam = true;
      } else if (options.reparam && !(dist.base && dist.transform)) {
        throw dist + ' does not support reparameterization.';
      } else {
        val = dist.sample();
        reparam = false;
      }

      if (dist.isContinuous && (!dist.base || !dist.transform)) {
        var msg = 'Warning: Continuous distribution ' + dist.meta.name + ' does not support reparameterization.';
        util.warn(msg, true);
      }

      return {val: val, reparam: reparam};
    },

    factor: function(s, k, a, score, name) {
      if (!isFinite(ad.value(score))) {
        throw new Error('ELBO: factor score is not finite.');
      }
      var m = top(this.mapDataStack).multiplier;
      var node = new FactorNode(
          this.prevNode, score, m, this.opts.debugWeights);
      this.prevNode = node;
      this.nodes.push(node);
      return k(s);
    },

    mapDataFetch: function(data, opts, address) {

      var batchSize = opts.batchSize !== undefined ? opts.batchSize : data.length;
      var minBatchSize = _.isEmpty(data) ? 0 : 1;
      var maxBatchSize = data.length;

      if (!(util.isInteger(batchSize) &&
            minBatchSize <= batchSize &&
            batchSize <= maxBatchSize)) {
        throw new Error('ELBO: Invalid batchSize.');
      }

      // Compute batch indices.

      var ix;
      if (_.has(this.mapDataIx, address)) {
        ix = this.mapDataIx[address];
      } else {
        if (batchSize === data.length) {
          // Use all the data, in order.
          ix = null;
        } else {
          ix = _.times(batchSize, function() {
            return Math.floor(util.random() * data.length);
          });
        }
        // Store batch indices so that we can use the same mini-batch
        // across samples.
        this.mapDataIx[address] = ix;
      }

      var batch = (ix === null) ? data : _.at(data, ix);

      if (batchSize > 0) {
        var joinNode = new JoinNode();
        var splitNode = new SplitNode(this.prevNode, batchSize, data.length, joinNode);
        this.nodes.push(splitNode);

        // Compute the multiplier required to account for the fact we're
        // only looking at a subset of the data.
        var thisM = data.length / batchSize;
        var prevM = top(this.mapDataStack).multiplier;
        var multiplier = thisM * prevM;

        this.mapDataStack.push({
          splitNode: splitNode,
          joinNode: joinNode,
          multiplier: multiplier
        });
      } else {
        // Signal to mapDataFinal that the batch was empty.
        this.mapDataStack.push(null);
      }

      return {data: batch, ix: ix};
    },

    mapDataEnter: function() {
      // For every observation function, set the current node back to
      // the split node.
      this.prevNode = top(this.mapDataStack).splitNode;
    },

    mapDataLeave: function() {
      // Hook-up the join node to the last node on this branch. If
      // there were no sample/factor nodes created in the observation
      // function then this connects the join node directly to the
      // split node. The correction applied to split nodes in
      // propagateWeights requires such edges to be present for
      // correctness.
      top(this.mapDataStack).joinNode.parents.push(this.prevNode);
    },

    mapDataFinal: function(address) {
      var top = this.mapDataStack.pop();
      if (top !== null) {
        var joinNode = top.joinNode;
        this.prevNode = joinNode;
        this.nodes.push(joinNode);
      }
    },

    incrementalize: env.defaultCoroutine.incrementalize,
    constructor: ELBO

  };

  return makeELBOEstimator;

};
