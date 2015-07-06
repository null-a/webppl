'use strict';

var _ = require('underscore');
var util = require('../util.js');
var erp = require('../erp.js');
var Trace = require('../trace.js').Trace;

// This is a stripped down particle filer. Is doesn't handle variable numbers of
// factors and uses a basic resampling strategy.

module.exports = function(env) {

  var MHKernel = require('./mhkernel')(env).MHKernel;

  function ParticleFilter(s, k, a, wpplFn, numParticles, rejuvSteps) {

    this.particles = [];
    this.particleIndex = 0;
    this.numParticles = numParticles;
    this.rejuvSteps = rejuvSteps;

    this.hist = {};

    var exitK = function(s) {
      return wpplFn(s, env.exit, a);
    };

    // Create initial particles.
    // Particles are partial/incomplete traces.
    for (var i = 0; i < numParticles; i++) {
      var p = new Trace();
      p.saveContinuation(exitK, _.clone(s));
      this.particles.push(p);
    }

    this.k = k;
    this.s = s;
    this.a = a;
    this.coroutine = env.coroutine;
    env.coroutine = this;

  }

  ParticleFilter.prototype.run = function() {
    return this.runCurrentParticle();
  };

  ParticleFilter.prototype.sample = function(s, k, a, erp, params) {
    var val = erp.sample(params);
    var choiceScore = erp.score(params, val);
    var particle = this.currentParticle();
    particle.addChoice(erp, params, val, choiceScore, a, s, k);
    return k(s, val);
  };

  ParticleFilter.prototype.factor = function(s, cc, a, score) {
    // Update particle.
    var particle = this.currentParticle();
    particle.saveContinuation(cc, s);
    particle.score += score;
    particle.weight = score; // Importance weights for resampling.

    var cont = function() {
      this.nextParticle();
      return this.runCurrentParticle();
    }.bind(this);

    // Resample/rejuvenate at the last particle.
    if (this.lastParticle()) {
      this.resampleParticles();
      return this.rejuvenateParticles(cont, a);
    } else {
      return cont();
    }
  };

  ParticleFilter.prototype.lastParticle = function() {
    return this.particleIndex === this.numParticles - 1;
  };

  ParticleFilter.prototype.currentParticle = function() {
    return this.particles[this.particleIndex];
  };

  ParticleFilter.prototype.nextParticle = function() {
    this.particleIndex = (this.particleIndex + 1) % this.numParticles;
  };

  ParticleFilter.prototype.runCurrentParticle = function() {
    return this.currentParticle().k(this.currentParticle().store);
  };

  var choose = function(ps) {
    // ps is expected to be normalized.
    var x = Math.random();
    var acc = 0;
    for (var i = 0; i < ps.length; i++) {
      acc += ps[i];
      if (x < acc) return i;
    }
    throw 'unreachable';
  };

  ParticleFilter.prototype.resampleParticles = function() {
    var ws = _.map(this.particles, function(p) { return Math.exp(p.weight); });
    var wsum = util.sum(ws);
    var wsnorm = _.map(ws, function(w) { return w / wsum; });

    assert(_.some(wsnorm, function(w) { return w > 0; }), 'No +ve weights: ' + ws);
    assert(_.every(wsnorm, function(w) { return w >= 0; }));

    this.particles = _.chain(_.range(this.numParticles))
      .map(function() {
          var ix = choose(wsnorm);
          assert(ix >= 0 && ix < this.numParticles);
          return this.particles[ix].copy();
        }.bind(this)).value()
  };

  // TODO: How can this be written in a more straight-foward way.

  // TODO: k here isn't a webppl continuation, rather it's a thunk, created in
  // factor. This doesn't need to be called with arguments and I don't think I
  // need to pass s & a around either.

  ParticleFilter.prototype.rejuvenateParticles = function(cont, exitAddress) {
    return util.cpsForEach(
        function(p, i, ps, next) {
          return this.rejuvenateParticle(next, i, exitAddress);
        }.bind(this),
        cont,
        this.particles
    );
  };

  ParticleFilter.prototype.rejuvenateParticle = function(cont, i, exitAddress) {

    // TODO: Check this is correct.

    // My intention is to run wpplFn with the same address as was used for the
    // particle filter so that addresses line up correctly with the trace. I'm
    // not sure I need to do this since the MHKernel will pick continue using
    // the address of an entry in the trace.

    // transition :: wpplFn x trace -> trace
    var transition = _.partial(MHKernel, this.s, _, this.a, this.wpplFn, _, exitAddress);

    var particle = this.particles[i];

    // TODO: This is similar to MCMC with initialization. Extract?

    return util.cpsLoop(this.rejuvSteps,
        function(j, next) {
          //console.log('Step: ' + j);
          return transition(function(s, newParticle) {
            particle = newParticle;
            return next();
          }, particle);
        },
        cont
    );
  };

  ParticleFilter.prototype.exit = function(s, val) {
    // Complete the trace.
    var particle = this.currentParticle();
    particle.complete(val);

    // Update histogram.
    var k = JSON.stringify(val);
    if (this.hist[k] === undefined) this.hist[k] = { prob: 0, val: val };
    this.hist[k].prob += 1;

    // Run any remaining particles.
    if (!this.lastParticle()) {
      this.nextParticle();
      return this.runCurrentParticle();
    }

    // Finished, call original continuation.
    var dist = erp.makeMarginalERP(this.hist);
    env.coroutine = this.coroutine;
    return this.k(this.s, dist);
  };


  return {
    ParticleFilter2: function(s, cc, a, wpplFn, numParticles, rejuvSteps) {
      return new ParticleFilter(s, cc, a, wpplFn, numParticles, rejuvSteps).run();
    }
  };

};
