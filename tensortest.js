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


var test2 = function() {

  var a = ad.lift(new Tensor([2, 1]).fromFlatArray([1, 2]));
  var b = ad.lift(new Tensor([2, 1]).fromFlatArray([3, 4]));
  var y = ad.tensor.sumreduce(ad.tensor.mul(ad.tensor.mul(a, a), b));

  y.backprop();

  console.log(ad.value(y));
  console.log(ad.derivative(a));
  console.log(ad.derivative(b));

};

var test4 = function() {

  var a = (new Tensor([3, 1]).fromFlatArray([10, 20, 30]));
  //var b = ad.lift(new Tensor([3, 1]).fromFlatArray([1, 2, 3]));

  var b = ad.lift(3);
  var y = ad.tensor.sumreduce(ad.tensor.sub(a, b));
  y.backprop();

  console.log(ad.value(y));
  console.log(ad.derivative(a));
  console.log(ad.derivative(b));

};

var test5 = function() {

  var a = ad.lift(new Tensor([2, 1]).fromFlatArray([10, 20]));
  //var b = ad.lift(new Tensor([2, 1]).fromFlatArray([1, 2]));
  var y = ad.tensor.sumreduce(ad.tensor.dot(ad.tensor.transpose(a), ad.tensor.neg(a)));
  y.backprop();


  console.log(ad.value(y));
  console.log(ad.derivative(a));
  //console.log(ad.derivative(b));



};

test5();
