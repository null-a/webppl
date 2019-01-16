'use strict';

var tf = require('../tf');

function isTensor(a) {
  return a instanceof tf.Tensor;
}

function neg(a) {
  if (isTensor(a)) {
    return tf.neg(a);
  }
  else {
    return -a;
  }
};

function log(a) {
  if (isTensor(a)) {
    return tf.log(a);
  }
  else {
    return Math.log(a);
  }
};

function exp(a) {
  if (isTensor(a)) {
    return tf.exp(a);
  }
  else {
    return Math.exp(a);
  }
};

function ceil(a) {
  if (isTensor(a)) {
    return tf.ceil(a);
  }
  else {
    return Math.ceil(a);
  }
};

function floor(a) {
  if (isTensor(a)) {
    return tf.floor(a);
  }
  else {
    return Math.floor(a);
  }
};

function mul(a, b) {
  if (isTensor(a) || isTensor(b)) {
    return tf.mul(a, b);
  }
  else {
    return a * b;
  }
};

function div(a, b) {
  if (isTensor(a) || isTensor(b)) {
    return tf.div(a, b);
  }
  else {
    return a / b;
  }
};

function add(a, b) {
  if (isTensor(a) || isTensor(b)) {
    return tf.add(a, b);
  }
  else {
    return a + b;
  }
};

function sub(a, b) {
  if (isTensor(a) || isTensor(b)) {
    return tf.sub(a, b);
  }
  else {
    return a - b;
  }
};

module.exports = {
  neg,
  log,
  exp,
  ceil,
  floor,
  mul,
  div,
  add,
  sub
};
