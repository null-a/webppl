'use strict';

var _ = require('lodash');
//var ad = require('../ad');
var tf = require('../tf');
var Tensor = require('../tensor');
var util = require('../util');
var tensorGaussian = require('../dists/tensorGaussian');
var config = require('./config');
var params = require('./params');
var serialize = require('./serialize');


function getParams(s, k, a) {
  return k(s, params.get());  // params.get is not a cps function
}

function setParams(s, k, a, prms) {
  return params.set(prms, function() { return k(s); });
}

function setParamsId(s, k, a, id) {
  config.setId(id);
  return params.sync(function() {
    return k(s, id);
  });
}

function setFreshParamsId(s, k, a) {
  var id = config.setFreshId();
  return params.sync(function() {
    return k(s, id);
  });
}

function getParamsId(s, k, a, id) {
  return k(s, config.getId());
}

function serializeParams(s, k, a, paramsObj) {
  return k(s, serialize.serializeParams(paramsObj));
}

function deserializeParams(s, k, a, str) {
  return k(s, serialize.deserializeParams(str));
}

function defaultInit(mu, sigma) {
  return function(s, k, a, dims) {
    return k(s, tensorGaussian.sample(mu, sigma, dims));
  };
}

module.exports = function(env) {

  var forward = require('../inference/forwardSample')(env).forward;

  var dimsForScalarParam = []; // no longer [1], since in tf.js scalars are rank 0 tensors

  var param = function(s, k, a, options) {
    options = util.mergeDefaults(options, {
      mu: 0,
      sigma: .1,
      dims: dimsForScalarParam
    });

    if (!env.coroutine._guide) {
      util.warn('Warning: Parameter ' +
                (_.has(options, 'name') ? '"' + options.name + '" ' : '') +
                'created outside of the guide.', true);
    }

    var dims = options.dims;
    var name = _.has(options, 'name') ?
        options.name :
        util.relativizeAddress(params.baseAddress(env), a);

    if (params.exists(name)) {
      return finish(s);
    } else {
      var init = _.has(options, 'init') ? options.init : defaultInit(options.mu, options.sigma);
      if (!_.isFunction(init)) {
        throw new Error('Expected the init argument to be a function.');
      }

      var initThunk = function(s, k, a) {
        return init(s, k, a, dims);
      };

      var next = function(k, initialVal) {
        params.create(name, initialVal);
        if (!_.isEqual(dims, initialVal.shape)) {
          var msg = 'The init function did not return a tensor with the expected shape.';
          throw new Error(msg);
        }
        return finish(s);
      };

      return forward(s, next, a, initThunk);
    }

    function finish(s) {
      var val = params.fetch(name, env);
      var valDims = val.shape; // ad.value(val).dims;
      if (!_.isEqual(dims, valDims)) {
        var msg = 'The dims specified here (' + JSON.stringify(dims) +
            ') do not match the dims of the current parameter value (' +
            JSON.stringify(valDims) + '). This value may ' +
            'come from an earlier call to param, or from a previous ' +
            'execution when a persistent parameter store is used.';
        throw new Error(msg);
      }
      // in adnn this extracts a (potentially lifted) scalar from a
      // tensor. under tf.js, in the setting where `fetch` has
      // "lifted" the parameter, we need to return a rank 0 tensor
      // (and not extract a js number) so that ad continues to work.
      // there's probably a choice to be made about what to do when
      // lifting is not required. contune to return a rank 0 tensor
      // for consistency, or extract scalar for (potential -- and it's
      // not clear there is any) convenience.
      //return k(s, dims === dimsForScalarParam ? ad.tensor.get(val, 0) : val);
      return k(s, val);
    }
  };

  return {
    getParams: getParams,
    getParamsId: getParamsId,
    param: param,
    setFreshParamsId: setFreshParamsId,
    setParams: setParams,
    setParamsId: setParamsId,
    serializeParams: serializeParams,
    deserializeParams: deserializeParams
  };
};
