'use strict';

var ad = require('./src/ad');
var Tensor = require('./src/tensor');

var test1 = function() {

  var X = ad.lift(new Tensor([2, 2]).fromArray([[2, 5], [.5, 1]]));

  // Y = log det X.T.dot(X)
  // Actual grad w.r.t. X:
  // 2 * pinv(X).T

  // Which at [[2, 5], [.5, 1]]
  // is:
  // [[-4, 2], [20, -8]]

  var Y = ad.scalar.log(ad.tensor.det(ad.tensor.dot(ad.tensor.transpose(X), X)));
  Y.backprop();
  console.log(ad.value(Y));
  console.log(ad.derivative(X));
};

test1();
