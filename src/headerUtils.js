'use strict';

var makeMarginalERP = require('./erp.js').makeMarginalERP;

module.exports = function(env) {

  function display(s, k, a, x) {
    return k(s, console.log(x));
  }

  // Caching for a wppl function f.
  //
  // Caution: if f isn't deterministic weird stuff can happen, since
  // caching is across all uses of f, even in different execuation
  // paths.
  function cache(s, k, a, f) {
    var c = {};
    var cf = function(s, k, a) {
      var args = Array.prototype.slice.call(arguments, 3);
      var stringedArgs = JSON.stringify(args);
      if (stringedArgs in c) {
        return k(s, c[stringedArgs]);
      } else {
        var newk = function(s, r) {
          c[stringedArgs] = r;
          return k(s, r);
        };
        return f.apply(this, [s, newk, a].concat(args));
      }
    };
    return k(s, cf);
  }

  function apply(s, k, a, wpplFn, args) {
    return wpplFn.apply(global, [s, k, a].concat(args));
  }

  // TODO: Is there a better way to use this than wrapping it?
  var wpplMakeMarginalERP = function(s, k, a, marginal) {
    return k(s, makeMarginalERP(marginal));
  };

  // TODO: Do I need to extend the stack address here?
  var runWithCoroutine = function(s, k, a, coroutine, wpplFn) {
    // Store the current coroutine.
    var entryCoroutine = env.coroutine;
    // Install the new coroutine.
    env.coroutine = coroutine;
    // Run the wpplFn with the coroutine.
    return coroutine.run(s, function(s, val) {
      // Restore the original coroutine.
      env.coroutine = entryCoroutine;
      return k(s, val);
    }, a, wpplFn);
  };

  // TODO: Do I need to extend the stack address here?
  var callcc = function(s, k, a, f) {
    return f(s, k, a, function(s, _, a, x) {
      return k(s, x)
    });
  };

  var getStore = function(s, k, a) {
    return k(s, s);
  };

  var setStore = function(s, k, a, store) {
    return k(store);
  };

  // Delimited continuations state and methods to manipulated it.

  env.metaContinuation = function() {
    throw 'No top-level reset.';
  };

  var getMeta = function(s, k, a) {
    return k(s, env.metaContinuation);
  };

  var setMeta = function(s, k, a, value) {
    env.metaContinuation = value;
    return k(s);
  };

  return {
    display: display,
    cache: cache,
    apply: apply,
    wpplMakeMarginalERP: wpplMakeMarginalERP,
    runWithCoroutine: runWithCoroutine,
    callcc: callcc,
    getStore: getStore,
    setStore: setStore,
    getMeta: getMeta,
    setMeta: setMeta
  };

};
